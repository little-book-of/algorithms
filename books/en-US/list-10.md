# Chapter 10. AI, ML and Optimization 

# Section 91. Classical ML 

### 901. k-Means Clustering

k-Means clustering is one of the simplest and most popular unsupervised learning algorithms. It groups data into *k* clusters so that each point belongs to the cluster with the nearest centroid. You can think of it as finding "gravity centers" that pull similar points together.

#### What Problem Are We Solving?

We often have raw data without labels and want to discover natural groupings.  
k-Means answers: *"Which points belong together?"*

Given:

- A dataset of points $X = \{x_1, x_2, \dots, x_n\}$
- A target number of clusters $k$

We want to find $k$ centroids $\mu_1, \mu_2, \dots, \mu_k$ that minimize total squared distance:

$$
\text{Objective: } \min_{\mu} \sum_{i=1}^{n} \lVert x_i - \mu_{c(i)} \rVert^2
$$

where $c(i)$ is the cluster assignment of point $x_i$.


#### How Does It Work (Plain Language)?

Think of *k* seeds dropped into your data. Each point chooses its nearest seed. Then the seeds move to the center of their assigned points. Repeat until the seeds stop moving. That's k-Means.

Step-by-step:

| Step | Action          | Description                               |
| ---- | --------------- | ----------------------------------------- |
| 1    | Initialize  | Pick *k* centroids (random or smart init) |
| 2    | Assign step | Assign each point to nearest centroid     |
| 3    | Update step | Move each centroid to mean of its points  |
| 4    | Repeat      | Until centroids stop changing (converge)  |

Example (2 clusters):

| Iteration | Centroid 1              | Centroid 2              | Description     |
| --------- | ----------------------- | ----------------------- | --------------- |
| 0         | Random points           | Random points           | Start anywhere  |
| 1         | Move to mean of cluster | Move to mean of cluster | Clusters adjust |
| 2         | Minimal change          | Minimal change          | Converged       |

#### Tiny Code (Easy Versions)

Python

```python
import numpy as np

def kmeans(X, k, iters=100):
    n = X.shape[0]
    centroids = X[np.random.choice(n, k, replace=False)]
    for _ in range(iters):
        distances = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(k)])
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return centroids, labels
```

C (Outline)

```c
// Simple pseudocode-style structure
for (iter = 0; iter < MAX_ITER; iter++) {
    assign_points_to_nearest_centroid();
    recompute_centroids_as_means();
    if (centroids_converged()) break;
}
```

#### Why It Matters

- Reveals hidden structure in unlabeled data.
- Foundation for clustering, image compression, and vector quantization.
- Builds intuition for iterative optimization (EM, Lloyd's algorithm).
- Runs fast, easy to implement, widely used as a baseline.

#### A Gentle Proof (Why It Works)

Each iteration decreases the sum of squared distances.

- Assign step: choosing the nearest centroid never increases cost.
- Update step: moving to the mean minimizes squared error.

Since there are finitely many clusterings, the algorithm must converge (though not necessarily to the *global* optimum).

Objective function:
$$
J = \sum_{i=1}^n | x_i - \mu_{c(i)} |^2
$$

At each step: ( J_{\text{new}} \le J_{\text{old}} )

#### Try It Yourself

1. Run on 2D points: ((1,1), (1.5,2), (5,8), (8,8)) with (k=2).
2. Try random vs. k-Means++ initialization.
3. Plot clusters and centroids after convergence.
4. Increase (k): what happens to cluster sizes?
5. Compare with hierarchical clustering results.

#### Test Cases

| Dataset              | k | Expected Behavior                     |
| -------------------- | - | ------------------------------------- |
| 4 points in 2 groups | 2 | Splits into 2 clusters                |
| All points in a line | 3 | Divides into segments                 |
| Identical points     | 2 | Both centroids converge to same point |
| Random cloud         | 3 | Forms roughly equal partitions        |

#### Complexity

- Time: ( O(n \times k \times d \times t) )
  (n points, k clusters, d dimensions, t iterations)
- Space: ( O(n + k) )

k-Means clustering is your lens for structure, a simple loop of "assign and update" that uncovers patterns where none were labeled.

### 902. k-Medoids (PAM)

k-Medoids clustering is like k-Means, but instead of using the *mean* of points as a center, it chooses actual data points (medoids) as the representatives of clusters. This makes it more robust to outliers and non-Euclidean distances.

#### What Problem Are We Solving?

Sometimes the "mean" of points doesn't make sense, especially when:

- The data has outliers that distort averages.
- Distances are not Euclidean (e.g. edit distance, Manhattan).
- We want cluster centers to be *real points* in the dataset.

k-Medoids solves this by picking *actual* examples as centers (medoids) to minimize total dissimilarity:

$$
\text{Objective: } \min_{M} \sum_{i=1}^n d(x_i, m_{c(i)})
$$

where ( M = {m_1, \dots, m_k} ) are medoids, and ( d(\cdot,\cdot) ) is any distance metric.

#### How Does It Work (Plain Language)?

Think of k-Medoids as "find the most central representative."
Each cluster picks one of its points that minimizes total distance to all others.

Step-by-step (PAM: Partitioning Around Medoids):

| Step | Action         | Description                                           |
| ---- | -------------- | ----------------------------------------------------- |
| 1    | Initialize | Pick *k* random points as medoids                     |
| 2    | Assign     | Assign each point to nearest medoid                   |
| 3    | Swap       | For each non-medoid point, try swapping with a medoid |
| 4    | Evaluate   | If swap reduces total cost, accept it                 |
| 5    | Repeat     | Until no better swap found (converged)                |

Example (k=2):

| Iteration | Medoids                       | Description           |
| --------- | ----------------------------- | --------------------- |
| 0         | Randomly select 2 points      | Start anywhere        |
| 1         | Reassign clusters, test swaps | Reduce total distance |
| 2         | No better swaps               | Stop                  |

#### Tiny Code (Easy Versions)

Python (Simplified PAM)

```python
import numpy as np

def pam(X, k, dist_fn):
    n = len(X)
    medoids = np.random.choice(n, k, replace=False)
    while True:
        # Assign points to nearest medoid
        distances = np.array([[dist_fn(X[i], X[m]) for m in medoids] for i in range(n)])
        labels = np.argmin(distances, axis=1)
        # Try swapping
        improved = False
        for i in range(n):
            if i in medoids: continue
            for m in medoids:
                new_medoids = medoids.copy()
                new_medoids[new_medoids == m] = i
                new_cost = sum(dist_fn(X[j], X[new_medoids[labels[j]]]) for j in range(n))
                old_cost = sum(dist_fn(X[j], X[medoids[labels[j]]]) for j in range(n))
                if new_cost < old_cost:
                    medoids = new_medoids
                    improved = True
        if not improved:
            break
    return medoids, labels
```

#### Why It Matters

- Robust to noise and outliers, medoids aren't pulled by extremes.
- Works with any distance metric (Euclidean, cosine, edit distance).
- Used in bioinformatics, text clustering, and anomaly detection.
- A great conceptual bridge from k-Means to more flexible clustering.

#### A Gentle Proof (Why It Works)

Each swap is guaranteed to not increase total cost:
$$
J = \sum_{i=1}^{n} d(x_i, m_{c(i)})
$$
Since there's a finite number of possible medoid sets, and each iteration strictly improves or preserves cost, the algorithm converges to a local minimum.

Unlike k-Means, no averaging is required, only distance comparisons.

#### Try It Yourself

1. Run k-Medoids with Manhattan distance on 2D points.
2. Add an outlier, see if medoids resist drift.
3. Compare results with k-Means (same *k*).
4. Test on string data (e.g. Levenshtein distance).
5. Visualize medoids as "chosen representatives".

#### Test Cases

| Dataset                       | k | Metric        | Expected Behavior                        |
| ----------------------------- | - | ------------- | ---------------------------------------- |
| Points with outlier           | 2 | Euclidean     | Medoid ignores outlier                   |
| Strings ("cat", "bat", "rat") | 2 | Edit distance | Clusters by similarity                   |
| Line of points                | 3 | Manhattan     | Centers are actual data points           |
| Random scatter                | 2 | Euclidean     | Partitions like k-Means but medoid-based |

#### Complexity

- Time: ( O(k (n-k)^2) ) (PAM)
- Space: ( O(n) )

k-Medoids clustering finds *real exemplars*, not averages, a method that stays true to your data and stands firm against outliers.

### 903. Gaussian Mixture Model (EM)

Gaussian Mixture Models (GMMs) take clustering to the probabilistic world. Instead of assigning each point to a single cluster, they let every point *belong to multiple clusters with different probabilities*. The model assumes data is generated from a mix of several Gaussian distributions.

#### What Problem Are We Solving?

Sometimes clusters overlap or have fuzzy boundaries. Hard assignments (like in k-Means) can be misleading.
We want soft clustering, each data point has a probability of belonging to each cluster.

Given:

- Data points ( X = {x_1, x_2, \dots, x_n} )
- A chosen number of components ( k )

We model:
$$
p(x) = \sum_{j=1}^k \pi_j , \mathcal{N}(x \mid \mu_j, \Sigma_j)
$$
where:

- ( \pi_j ): weight (mixture proportion)
- ( \mu_j ): mean of component ( j )
- ( \Sigma_j ): covariance of component ( j )

#### How Does It Work (Plain Language)?

Think of GMM as "soft k-Means."
Each point is not assigned to one cluster, but gets fractional membership, "70% cluster A, 30% cluster B."

We use the Expectation–Maximization (EM) algorithm to estimate parameters.

| Step | Name                      | Description                                         |
| ---- | ------------------------- | --------------------------------------------------- |
| 1    | Initialize            | Pick ( \mu_j, \Sigma_j, \pi_j )                     |
| 2    | E-step (Expectation)  | Compute membership probabilities (responsibilities) |
| 3    | M-step (Maximization) | Update parameters using weighted averages           |
| 4    | Repeat                | Until convergence (log-likelihood stabilizes)       |

E-step formula:
$$
\gamma_{ij} = \frac{\pi_j \mathcal{N}(x_i \mid \mu_j, \Sigma_j)}{\sum_{l=1}^k \pi_l \mathcal{N}(x_i \mid \mu_l, \Sigma_l)}
$$

M-step updates:
$$
\mu_j = \frac{\sum_i \gamma_{ij} x_i}{\sum_i \gamma_{ij}}, \quad
\Sigma_j = \frac{\sum_i \gamma_{ij} (x_i - \mu_j)(x_i - \mu_j)^T}{\sum_i \gamma_{ij}}, \quad
\pi_j = \frac{1}{n} \sum_i \gamma_{ij}
$$

#### Tiny Code (Easy Versions)

Python (with NumPy, minimal EM)

```python
import numpy as np
from scipy.stats import multivariate_normal

def gmm_em(X, k, iters=100):
    n, d = X.shape
    # Initialize
    np.random.seed(0)
    mu = X[np.random.choice(n, k, replace=False)]
    sigma = [np.eye(d)] * k
    pi = np.ones(k) / k
    
    for _ in range(iters):
        # E-step
        gamma = np.zeros((n, k))
        for j in range(k):
            gamma[:, j] = pi[j] * multivariate_normal.pdf(X, mu[j], sigma[j])
        gamma /= gamma.sum(axis=1, keepdims=True)
        
        # M-step
        Nk = gamma.sum(axis=0)
        for j in range(k):
            mu[j] = (gamma[:, j][:, None] * X).sum(axis=0) / Nk[j]
            x_centered = X - mu[j]
            sigma[j] = (gamma[:, j][:, None, None] * 
                        np.einsum('ni,nj->nij', x_centered, x_centered)).sum(axis=0) / Nk[j]
            pi[j] = Nk[j] / n
    return mu, sigma, pi
```

#### Why It Matters

- Handles overlapping clusters naturally.
- Produces probabilistic assignments instead of hard labels.
- Supports elliptical (not just spherical) clusters via covariance.
- Foundation for EM algorithm, soft clustering, and latent variable models.
- Used in speech recognition, image segmentation, and density estimation.

#### A Gentle Proof (Why It Works)

EM alternates between:

- E-step: Estimate hidden variables (responsibilities).
- M-step: Maximize likelihood given responsibilities.

Each iteration does not decrease the data log-likelihood:
$$
\log p(X \mid \theta) = \sum_{i=1}^{n} \log \sum_{j=1}^{k} \pi_j \mathcal{N}(x_i | \mu_j, \Sigma_j)
$$
Hence, convergence is guaranteed (though possibly to a local maximum).

#### Try It Yourself

1. Cluster 2D points with overlapping Gaussians.
2. Visualize ellipses ((\Sigma_j)) to see cluster shapes.
3. Compare GMM clusters vs. k-Means on same data.
4. Change (k): see underfitting vs. overfitting.
5. Add small noise, GMM handles it better than k-Means.

#### Test Cases

| Dataset                | k | Behavior                   |
| ---------------------- | - | -------------------------- |
| 2D blobs (overlap)     | 2 | Smooth boundaries          |
| Non-spherical clusters | 3 | Elliptical shapes          |
| Well-separated data    | 2 | Matches k-Means            |
| Single Gaussian        | 1 | Learns mean and covariance |

#### Complexity

- Time: ( O(nkd) ) per iteration
- Space: ( O(nk) ) (responsibilities)

Gaussian Mixture Models blend geometry with probability, instead of forcing points into boxes, they let them *belong* where they most likely fit.

### 904. Naive Bayes Classifier

Naive Bayes is a simple yet powerful probabilistic classifier based on Bayes' theorem with a bold assumption: all features are conditionally independent given the class. Despite the "naive" assumption, it works surprisingly well across text, spam filtering, and many real-world tasks.

#### What Problem Are We Solving?

We want to predict a class label from a set of features, using probabilities rather than distances or hyperplanes.

Given:

- Training data ((x_i, y_i))
- Features (x = (x_1, x_2, \dots, x_d))
- Labels (y \in {1, 2, \dots, K})

We want:
$$
\hat{y} = \arg\max_y P(y \mid x)
$$

By Bayes' theorem:
$$
P(y \mid x) = \frac{P(x \mid y) P(y)}{P(x)}
$$

Since (P(x)) is constant across classes:
$$
\hat{y} = \arg\max_y P(y) , P(x \mid y)
$$

With feature independence:
$$
P(x \mid y) = \prod_{j=1}^d P(x_j \mid y)
$$

So we only need per-feature class conditionals.

#### How Does It Work (Plain Language)?

Naive Bayes looks at each feature separately, multiplies how likely each one is under a class, and picks the class with the biggest combined probability.

Step-by-step:

| Step | Action       | Description                                             |
| ---- | ------------ | ------------------------------------------------------- |
| 1    | Count    | Estimate (P(y)) = class frequency                       |
| 2    | Estimate | For each feature, estimate (P(x_j \mid y))              |
| 3    | Predict  | For a new example, compute (P(y) \prod_j P(x_j \mid y)) |
| 4    | Select   | Choose class with highest probability                   |

Example (spam filtering):

- Class: spam / not spam
- Features: words like "buy", "free", "click"
- Each word adds weight to spam probability.

#### Tiny Code (Easy Versions)

Python (Discrete Naive Bayes)

```python
from collections import defaultdict, Counter

class NaiveBayes:
    def __init__(self):
        self.class_counts = Counter()
        self.feature_counts = defaultdict(Counter)
        self.total = 0

    def fit(self, X, y):
        self.total = len(y)
        for xi, yi in zip(X, y):
            self.class_counts[yi] += 1
            for f in xi:
                self.feature_counts[yi][f] += 1

    def predict(self, x):
        scores = {}
        for c in self.class_counts:
            log_prob = 0
            for f in x:
                count = self.feature_counts[c][f]
                total = sum(self.feature_counts[c].values())
                log_prob += np.log((count + 1) / (total + len(self.feature_counts[c])))
            prior = np.log(self.class_counts[c] / self.total)
            scores[c] = prior + log_prob
        return max(scores, key=scores.get)
```

#### Why It Matters

- Fast: simple counting, no iteration.
- Scalable: works great on large text corpora.
- Robust: handles high-dimensional sparse data.
- Interpretable: probabilities explain predictions.
- Versatile: variants handle continuous (Gaussian) or multinomial data.

Common variants:

| Type        | Use Case                               |
| ----------- | -------------------------------------- |
| Bernoulli   | Binary features (word presence)        |
| Multinomial | Word counts (text)                     |
| Gaussian    | Continuous data (e.g. sensor readings) |

#### A Gentle Proof (Why It Works)

From Bayes' theorem:
$$
P(y|x) \propto P(y) \prod_{j=1}^d P(x_j|y)
$$

The independence assumption simplifies joint probabilities into per-feature terms, reducing exponential complexity to linear.
Although independence is rarely true, the resulting classifier still performs well when relative likelihoods are correct, *a happy accident of probability algebra.*

#### Try It Yourself

1. Build a spam detector with "free", "offer", "hello".
2. Compute probabilities manually for small dataset.
3. Try Gaussian version on numeric features.
4. Compare accuracy vs. Logistic Regression.
5. Observe how Laplace smoothing affects zero counts.

#### Test Cases

| Dataset              | Variant     | Expected                 |
| -------------------- | ----------- | ------------------------ |
| Text spam detection  | Multinomial | Good accuracy            |
| Binary word features | Bernoulli   | Robust prediction        |
| Sensor data          | Gaussian    | Smooth decision boundary |
| Small dataset        | Any         | Needs smoothing          |

#### Complexity

- Training: ( O(nd) ) (counting)
- Prediction: ( O(kd) ) per example
- Space: ( O(kd) )

Naive Bayes is your probabilistic compass, simple counts and multiplications that turn uncertainty into decisions.

### 905. Logistic Regression

Logistic Regression is the workhorse of classification, a simple, powerful model that predicts probabilities instead of raw scores. It draws a decision boundary in feature space using the logistic (sigmoid) function, mapping linear combinations into the range (0, 1).

#### What Problem Are We Solving?

We want to predict a binary label $y \in \{0, 1\}$ from a feature vector $x \in \mathbb{R}^d$,  
where the output is a probability rather than a discrete class.

We model:
$$
P(y = 1 \mid x) = \sigma(w^\top x + b)
$$
where:
$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$
is the sigmoid function.

Decision rule:
$$
\hat{y} =
\begin{cases}
1, & \text{if } \sigma(w^\top x + b) > 0.5,\\
0, & \text{otherwise.}
\end{cases}
$$


We learn ( w, b ) by maximum likelihood estimation, equivalent to minimizing log loss:
$$
L(w, b) = -\frac{1}{n} \sum_{i=1}^{n} \big[ y_i \log \hat{p}_i + (1 - y_i) \log (1 - \hat{p}_i) \big]
$$

#### How Does It Work (Plain Language)?

Imagine fitting a smooth S-shaped curve that separates two classes. Instead of hard cutoffs, Logistic Regression gives *confidence* in each prediction.

Step-by-step:

| Step | Action           | Description                                     |
| ---- | ---------------- | ----------------------------------------------- |
| 1    | Initialize   | Start with random weights ( w, b )              |
| 2    | Predict      | Compute $\hat{p}_i = \sigma(w^\top x_i + b)$  |
| 3    | Compute Loss | Cross-entropy between predicted and true labels |
| 4    | Update       | Adjust weights using gradient descent           |
| 5    | Repeat       | Until loss converges                            |

Gradient update:
$$
w := w - \eta \frac{\partial L}{\partial w} = w - \eta (X^\top (\hat{p} - y))
$$
$$
b := b - \eta \sum_i (\hat{p}_i - y_i)
$$

#### Tiny Code (Easy Versions)

Python (from scratch)

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression(X, y, lr=0.1, epochs=1000):
    n, d = X.shape
    w = np.zeros(d)
    b = 0
    for _ in range(epochs):
        z = X @ w + b
        p = sigmoid(z)
        dw = (1/n) * X.T @ (p - y)
        db = (1/n) * np.sum(p - y)
        w -= lr * dw
        b -= lr * db
    return w, b

def predict(X, w, b):
    return (sigmoid(X @ w + b) >= 0.5).astype(int)
```

C (Outline)

```c
// For each iteration:
// 1. Compute z = w·x + b
// 2. Compute p = 1 / (1 + exp(-z))
// 3. Compute gradients dw, db
// 4. Update parameters
```

#### Why It Matters

- Interpretable: coefficients show feature influence.
- Probabilistic output: unlike hard-margin models.
- Foundation for neural networks (sigmoid neurons).
- Efficient for large-scale classification.
- Extensible: with regularization (L1, L2).

Variants:

| Type        | Description             |
| ----------- | ----------------------- |
| L1 (Lasso)  | Sparse weights          |
| L2 (Ridge)  | Smooth regularization   |
| Multinomial | Multi-class via softmax |

#### A Gentle Proof (Why It Works)

Logistic regression arises from maximum likelihood estimation for Bernoulli-distributed labels:
$$
P(y_i \mid x_i) = \hat{p}_i^{y_i} (1 - \hat{p}_i)^{1 - y_i}
$$

Maximizing likelihood is equivalent to minimizing log loss.
Gradient descent guarantees convergence to a convex global minimum, no local traps.

#### Try It Yourself

1. Create 2D dataset with two classes.
2. Plot sigmoid boundary and probabilities.
3. Add regularization, watch feature weights shrink.
4. Compare with Naive Bayes on same data.
5. Extend to multiclass with softmax.

#### Test Cases

| Dataset               | Expected Boundary      | Notes              |
| --------------------- | ---------------------- | ------------------ |
| Linearly separable    | Straight line          | Perfect separation |
| Overlapping classes   | Smooth transition      | Probabilistic      |
| High-dimensional text | Sparse weights         | L1 regularization  |
| Imbalanced classes    | Biased toward majority | Use class weights  |

#### Complexity

- Training: ( O(nd) ) per epoch
- Prediction: ( O(d) ) per example
- Space: ( O(d) )

Logistic Regression is your gateway from geometry to probability, drawing decision curves that *think in confidence, not absolutes*.

### 906. Perceptron

The Perceptron is one of the earliest and simplest models of a neuron, a linear classifier that learns by trial and error. It draws a hyperplane that separates two classes, adjusting its weights whenever it makes a mistake.

#### What Problem Are We Solving?

We want to find a linear boundary that separates two classes $y \in \{-1, +1\}$ given feature vectors $x \in \mathbb{R}^d$.

The Perceptron seeks weights $w$ and bias $b$ such that:

$$
y_i (w^\top x_i + b) > 0 \quad \forall i
$$

That means all positive examples lie on one side of the boundary, and all negative examples lie on the other.

#### How Does It Work (Plain Language)?

Imagine a line (or plane) that classifies points.  
If a point is misclassified, the Perceptron nudges the line toward it, step by step, until all points are correctly classified (if possible).

Step-by-step:

| Step | Action      | Description                                                        |
| ---- | ------------ | ------------------------------------------------------------------ |
| 1    | Initialize   | Set $w = 0$, $b = 0$                                               |
| 2    | Iterate      | For each training example:                                         |
| 3    | Predict      | $\hat{y} = \text{sign}(w^\top x + b)$                             |
| 4    | Update       | If wrong, adjust: $w \gets w + \eta y x$, $b \gets b + \eta y$    |
| 5    | Repeat       | Until no mistakes or max epochs                                   |

Each update shifts the boundary toward the misclassified point, improving alignment.


#### Tiny Code (Easy Versions)

Python

```python
import numpy as np

def perceptron(X, y, lr=1.0, epochs=100):
    n, d = X.shape
    w = np.zeros(d)
    b = 0
    for _ in range(epochs):
        errors = 0
        for xi, yi in zip(X, y):
            if yi * (np.dot(w, xi) + b) <= 0:
                w += lr * yi * xi
                b += lr * yi
                errors += 1
        if errors == 0:
            break
    return w, b

def predict(X, w, b):
    return np.sign(X @ w + b)
```

C (Outline)

```c
// For each (x_i, y_i):
//   if y_i * (dot(w, x_i) + b) <= 0:
//       w = w + lr * y_i * x_i
//       b = b + lr * y_i
```

#### Why It Matters

- Historical cornerstone of neural networks.
- Simple online learning: updates on mistakes only.
- Converges if data is linearly separable.
- Basis for modern models (SVM, SGDClassifier).

You can think of it as a "one-neuron brain", small, fast, and surprisingly effective when boundaries are linear.

#### A Gentle Proof (Why It Works)

If the data is linearly separable, the Perceptron converges in a finite number of steps.

Let the margin be:

$$
\gamma = \min_i \frac{y_i (w^* \cdot x_i)}{\lVert w^* \rVert}
$$

Then the number of updates $U$ satisfies:

$$
U \le \left(\frac{R}{\gamma}\right)^2
$$

where $R = \max_i \lVert x_i \rVert$.

Each mistake improves alignment with the true separator, so the algorithm cannot loop forever.

#### Try It Yourself

1. Train on 2D linearly separable data (for example, two distinct clusters).  
2. Visualize the decision boundary after each epoch.  
3. Flip a few labels to observe non-convergence.  
4. Try different learning rates such as $\eta = 0.1$ and $\eta = 1.0$.  
5. Compare the behavior with Logistic Regression on the same data.


#### Test Cases

| Dataset             | Expected Behavior               |
| ------------------- | ------------------------------- |
| Linearly separable  | Converges to correct hyperplane |
| Overlapping classes | Oscillates or never converges   |
| High dimension      | Learns if separable             |
| Random noise        | May not converge                |

#### Complexity

- Time: $O(nd \times \text{epochs})$
- Space: $O(d)$

The Perceptron is your first glimpse of a learning machine, a simple rule that sees mistakes, learns, and moves on.

### 907. Decision Tree (CART)

Decision Trees split data step by step, forming a hierarchy of decisions that classify or predict outcomes. Each node asks a yes/no question about a feature, dividing the data until it becomes pure, mostly one class or tightly clustered values.

#### What Problem Are We Solving?

We want a model that is:

- Interpretable — the reasoning is visible in the tree.  
- Flexible — handles numeric and categorical data.  
- Recursive — builds structure by partitioning data into smaller, simpler subsets.  

Given training data $(x_i, y_i)$, the goal is to find a sequence of splits that minimize impurity or variance.  
Formally, at each node we choose a feature $j$ and threshold $t$ to minimize:

$$
\min_{j, t} \Bigg( 
\frac{n_L}{n} \cdot \text{Impurity}(\text{left}) + 
\frac{n_R}{n} \cdot \text{Impurity}(\text{right}) 
\Bigg)
$$

where $n_L$ and $n_R$ are the sizes of the left and right subsets.

Common impurity measures for classification:

- Gini impurity  
  $$
  G = 1 - \sum_c p_c^2
  $$

- Entropy  
  $$
  H = -\sum_c p_c \log_2 p_c
  $$

For regression tasks, impurity is often measured by variance.


#### How Does It Work (Plain Language)?

Think of the tree as a flowchart. Each node asks a simple yes/no question,  
"Is feature $x_j \le t$?", and sends the sample left or right depending on the answer.  
The algorithm repeats recursively, choosing the best question at each node.

Step-by-step (CART algorithm):

| Step | Action        | Description                                                            |
| ---- | -------------- | ---------------------------------------------------------------------- |
| 1    | Start          | All data at the root node                                              |
| 2    | Search Splits  | For each feature $j$ and threshold $t$, compute impurity reduction     |
| 3    | Best Split     | Choose $(j, t)$ that maximizes impurity gain                           |
| 4    | Divide         | Split dataset into left/right subsets                                  |
| 5    | Repeat         | Recurse on each subset until stopping rule                             |
| 6    | Label Leaves   | Assign majority class or mean value                                    |

Stopping rules include:

- Maximum depth  
- Minimum number of samples per leaf  
- No impurity improvement


#### Tiny Code (Easy Versions)

Python (Simplified Binary Tree Split Finder)

```python
import numpy as np

def gini(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1 - np.sum(p  2)

def best_split(X, y):
    best_gain, best_j, best_t = 0, None, None
    base_impurity = gini(y)
    n, d = X.shape
    for j in range(d):
        thresholds = np.unique(X[:, j])
        for t in thresholds:
            left = y[X[:, j] <= t]
            right = y[X[:, j] > t]
            if len(left) == 0 or len(right) == 0:
                continue
            impurity = (len(left) * gini(left) + len(right) * gini(right)) / len(y)
            gain = base_impurity - impurity
            if gain > best_gain:
                best_gain, best_j, best_t = gain, j, t
    return best_j, best_t, best_gain
```

C (Outline)

```c
// For each node:
// 1. Compute impurity at the node
// 2. For each feature, test possible thresholds
// 3. Pick the split with highest impurity reduction
// 4. Recurse on left and right subsets
```

#### Why It Matters

- Interpretable: each split is a human-readable rule.
- Nonlinear boundaries: trees can represent piecewise decision regions.
- Versatile: supports both classification and regression.
- Scalable foundation: used in Random Forests and Gradient Boosting.
- No preprocessing: no need to normalize or scale features.

#### A Gentle Proof (Why It Works)

Each split reduces total impurity:

$$
\Delta = I_{\text{parent}} - \frac{n_L}{n} I_{\text{left}} - \frac{n_R}{n} I_{\text{right}}
$$

Because impurity measures ( I(\cdot) ) are non-negative and splitting always lowers impurity (or halts when no improvement), the algorithm eventually reaches a point where no further gain is possible. Thus, the tree converges to a structure with locally optimal splits.

#### Try It Yourself

1. Train a tree on a simple dataset (e.g. two clusters).
2. Print the rules: "if feature ≤ threshold, go left."
3. Limit the maximum depth to avoid overfitting.
4. Compare Gini vs Entropy for the same data.
5. Visualize decision boundaries in 2D.

#### Test Cases

| Dataset                   | Expected Behavior        |
| ------------------------- | ------------------------ |
| Linearly separable data   | Single root split        |
| Categorical + numeric mix | Handles both             |
| Overfitting example       | Deep tree memorizes data |
| Pruned tree               | Better generalization    |

#### Complexity

- Training: $O(n d \log n)$
- Prediction: $O(\text{depth})$
- Space: proportional to number of nodes

A Decision Tree is a divide-and-conquer learner, each question splits uncertainty, carving out regions of clarity until the data is neatly classified.

### 908. ID3 Algorithm

The ID3 (Iterative Dichotomiser 3) algorithm builds a decision tree using information gain, a measure of how much a feature helps reduce uncertainty (entropy). It's one of the earliest and most influential tree-learning algorithms, forming the foundation for later methods like C4.5 and CART.

#### What Problem Are We Solving?

We want to learn a classification tree that explains the data using informative splits.  
Each split should maximize reduction in entropy, making the resulting subsets as pure as possible.

Given a dataset $D$ with classes $C_1, C_2, \ldots, C_k$, the entropy of the set is:

$$
H(D) = - \sum_{c=1}^{k} p_c \log_2 p_c
$$

where $p_c$ is the proportion of samples in class $C_c$.

When we split $D$ by a feature $A$ with possible values $\{v_1, v_2, \dots, v_m\}$,  
the information gain is:

$$
\text{Gain}(D, A) = H(D) - \sum_{i=1}^{m} \frac{|D_{v_i}|}{|D|} \, H(D_{v_i})
$$

We choose the feature $A$ that maximizes $\text{Gain}(D, A)$.


#### How Does It Work (Plain Language)?

Think of ID3 as a "twenty questions" learner, each question (feature) splits the data to make it more predictable.
It always picks the most informative question first, then recurses.

Step-by-step:

| Step | Action                  | Description                                         |
| ---- | ----------------------- | --------------------------------------------------- |
| 1    | Compute Entropy     | Measure impurity of the current dataset             |
| 2    | For Each Feature    | Compute expected entropy after splitting            |
| 3    | Select Best Feature | Pick feature with maximum information gain          |
| 4    | Split               | Partition data by feature values                    |
| 5    | Recurse             | Build subtree for each subset                       |
| 6    | Stop                | If all samples have same class, or no features left |

#### Example

Suppose we have weather data:

| Outlook  | Temperature | Humidity | Wind   | Play |
| -------- | ----------- | -------- | ------ | ---- |
| Sunny    | Hot         | High     | Weak   | No   |
| Overcast | Cool        | Normal   | Strong | Yes  |
| Rain     | Mild        | High     | Weak   | Yes  |

ID3 computes entropy of `Play`, evaluates gain for each feature, and splits on the one with highest information gain (e.g., `Outlook`).

#### Tiny Code (Easy Versions)

Python (Simplified ID3)

```python
import numpy as np

def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return -np.sum(p * np.log2(p + 1e-9))

def info_gain(X_col, y):
    values, counts = np.unique(X_col, return_counts=True)
    weighted_entropy = 0
    for v, c in zip(values, counts):
        subset = y[X_col == v]
        weighted_entropy += (c / len(y)) * entropy(subset)
    return entropy(y) - weighted_entropy

def best_feature(X, y):
    gains = [info_gain(X[:, j], y) for j in range(X.shape[1])]
    return np.argmax(gains)
```

C (Outline)

```c
// 1. Compute entropy of current dataset
// 2. For each feature, partition by value and compute weighted entropy
// 3. Select feature with maximum gain
// 4. Recurse on subsets
```

#### Why It Matters

- Interpretable: produces readable decision rules.
- Greedy yet effective: picks best local split.
- Foundation: forms basis of C4.5 (handles continuous values, pruning).
- No assumptions: works directly from data frequencies.

ID3 was a major step in symbolic AI and machine learning, showing that *decision-making can be learned from data itself.*

#### A Gentle Proof (Why It Works)

Entropy measures average uncertainty:

$$
H(D) = - \sum_c p_c \log_2 p_c
$$

Splitting reduces entropy by creating subsets that are purer on average.  
Because $\text{Gain}(D, A) \ge 0$, each split either improves or leaves entropy unchanged.  
The recursion terminates when subsets are perfectly pure ($H = 0$) or no features remain.


#### Try It Yourself

1. Build a small dataset (like "Play Tennis").
2. Compute entropy by hand.
3. Evaluate information gain for each feature.
4. Draw the resulting tree.
5. Compare results with CART (Gini-based).

#### Test Cases

| Dataset            | Expected Behavior       |
| ------------------ | ----------------------- |
| Pure class labels  | Stop immediately        |
| Mixed features     | Choose most informative |
| Duplicate examples | Handle consistently     |
| Numeric features   | Requires discretization |

#### Complexity

- Training: $O(n d \log n)$ (depends on splits)
- Prediction: $O(\text{depth})$
- Space: proportional to number of nodes

The ID3 algorithm learns by asking the right questions first, each split is a move toward certainty, a step in building knowledge from entropy.

### 909. k-Nearest Neighbors (kNN)

The k-Nearest Neighbors (kNN) algorithm classifies a new point by looking at its closest examples in the training data. It's a simple, memory-based method: similar points tend to share the same label. Instead of learning parameters, kNN stores all training data and uses proximity to infer predictions.

#### What Problem Are We Solving?

We want a non-parametric way to perform classification or regression purely based on similarity.

Given:

- A training set $(x_1, y_1), \dots, (x_n, y_n)$
- A distance function $d(x, x')$
- A chosen number of neighbors $k$

The goal is to predict $\hat{y}$ for a new query point $x$.

For classification:

$$
\hat{y} = \arg\max_{c} \sum_{i \in \mathcal{N}_k(x)} \mathbf{1}(y_i = c)
$$

For regression:

$$
\hat{y} = \frac{1}{k} \sum_{i \in \mathcal{N}_k(x)} y_i
$$

where $\mathcal{N}_k(x)$ is the set of indices corresponding to the $k$ nearest neighbors of $x$.

#### How Does It Work (Plain Language)?

Imagine plotting your dataset on a plane. When a new point arrives, you measure how far it is from every known point, pick the $k$ closest, and use their labels to make a decision. The model doesn't generalize, it *remembers*.

Step-by-step:

| Step | Action                | Description                                            |
| ---- | --------------------- | ------------------------------------------------------ |
| 1    | Choose $k$        | Decide how many neighbors to consider                  |
| 2    | Compute Distances | Measure distance between query and all training points |
| 3    | Find Neighbors    | Select $k$ closest samples                             |
| 4    | Aggregate         | Majority vote (classification) or mean (regression)    |
| 5    | Predict           | Return the final label or value                        |

#### Example

Suppose $k = 3$:

- Nearest neighbors' labels: `[A, A, B]`
- Majority class: $A \Rightarrow \hat{y} = A$

For regression:

- Nearest neighbor values: $[4.0, 5.0, 3.0]$
- Prediction:

$$
\hat{y} = \frac{4.0 + 5.0 + 3.0}{3} = 4.0
$$

#### Tiny Code (Easy Versions)

Python (Simple kNN Classifier)

```python
import numpy as np
from collections import Counter

def knn_predict(X_train, y_train, x_query, k=3):
    distances = np.linalg.norm(X_train - x_query, axis=1)
    k_idx = np.argsort(distances)[:k]
    k_labels = y_train[k_idx]
    most_common = Counter(k_labels).most_common(1)
    return most_common[0][0]
```

C (Outline)

```c
// For each query point:
// 1. Compute distance to all training points
// 2. Sort and pick top-k neighbors
// 3. Take majority label (classification) or mean (regression)
```

#### Why It Matters

- Intuitive: classifies by similarity
- No training: all computation at query time
- Versatile: works for classification and regression
- Strong baseline: often a first model to compare others against

#### A Gentle Proof (Why It Works)

If $n \to \infty$, $k \to \infty$, and $\frac{k}{n} \to 0$,
then the kNN classifier approaches the Bayes optimal classifier, the theoretical best.

Reasoning:

- Neighbors approximate the local distribution near $x$
- Majority vote converges to the most probable class

Thus, kNN is statistically consistent under these conditions.

#### Try It Yourself

1. Generate a 2D dataset with two colored clusters
2. Try $k = 1, 3, 5$ and visualize decision boundaries
3. Add noise and increase $k$, observe smoothing
4. Experiment with distance metrics (Euclidean, Manhattan)
5. Compare results with a Decision Tree or Logistic Regression

#### Test Cases

| Dataset             | $k$   | Behavior                  |
| ------------------- | ----- | ------------------------- |
| Two clusters        | 1     | Sharp, irregular boundary |
| Two clusters        | 5     | Smooth boundary           |
| Noisy labels        | Large | More robust               |
| Overlapping classes | Small | Flexible but unstable     |

#### Complexity

- Training: $O(1)$ (lazy learner)
- Prediction: $O(n \cdot d)$ (compute all distances)
- Space: $O(n \cdot d)$

The k-Nearest Neighbors algorithm is a *memory of examples*: it learns nothing explicitly, but answers by looking around.

### 910. Linear Discriminant Analysis (LDA)

Linear Discriminant Analysis (LDA) is a probabilistic classifier that finds a linear boundary between classes by modeling each as a Gaussian distribution and assuming they share the same covariance. It combines geometry and probability, drawing decision lines where likelihoods balance.

#### What Problem Are We Solving?

We want to classify samples into $K$ classes by assuming:

1. Each class $C_k$ follows a Gaussian (normal) distribution
2. All classes share the same covariance matrix $\Sigma$

Given a point $x$, we choose the class with the highest posterior probability:

$$
\hat{y} = \arg\max_k ; P(C_k \mid x)
$$

By Bayes' theorem:

$$
P(C_k \mid x) \propto P(x \mid C_k) , P(C_k)
$$

Since $P(x)$ is constant across classes, we compare discriminant scores:

$$
\delta_k(x) = x^\top \Sigma^{-1} \mu_k - \frac{1}{2} \mu_k^\top \Sigma^{-1} \mu_k + \log P(C_k)
$$

The predicted class is the one with the largest $\delta_k(x)$.

#### How Does It Work (Plain Language)?

LDA imagines each class as a cloud shaped like an ellipse (Gaussian).
It computes how likely a new point is to come from each cloud, then picks the class with the highest likelihood.
Because covariances are shared, the boundaries between clouds are linear.

Step-by-step:

| Step | Action                         | Description                                                                              |
| ---- | ------------------------------ | ---------------------------------------------------------------------------------------- |
| 1    | Estimate Mean              | $\mu_k = \frac{1}{n_k} \sum_{i \in C_k} x_i$                                             |
| 2    | Estimate Shared Covariance | $\Sigma = \frac{1}{n - K} \sum_{k=1}^K \sum_{i \in C_k} (x_i - \mu_k)(x_i - \mu_k)^\top$ |
| 3    | Estimate Priors            | $P(C_k) = \frac{n_k}{n}$                                                                 |
| 4    | Compute Discriminant       | $\delta_k(x)$ for each class                                                             |
| 5    | Predict                    | Choose class with largest $\delta_k(x)$                                                  |

#### Example (2-Class Case)

When $K=2$, the decision boundary is a line:

$$
(w^\top x) + w_0 = 0
$$

where

$$
w = \Sigma^{-1}(\mu_1 - \mu_2), \quad
w_0 = -\frac{1}{2} (\mu_1 + \mu_2)^\top \Sigma^{-1}(\mu_1 - \mu_2) + \log\frac{P(C_1)}{P(C_2)}
$$

The sign of $(w^\top x + w_0)$ determines the predicted class.

#### Tiny Code (Easy Versions)

Python (2-Class LDA)

```python
import numpy as np

def lda_fit(X, y):
    classes = np.unique(y)
    n, d = X.shape
    means = {c: X[y == c].mean(axis=0) for c in classes}
    priors = {c: len(X[y == c]) / n for c in classes}
    # Shared covariance
    cov = np.zeros((d, d))
    for c in classes:
        Xc = X[y == c] - means[c]
        cov += Xc.T @ Xc
    cov /= (n - len(classes))
    inv_cov = np.linalg.inv(cov)
    return means, priors, inv_cov

def lda_predict(X, means, priors, inv_cov):
    scores = []
    for c in means:
        term = X @ inv_cov @ means[c]
        const = -0.5 * means[c].T @ inv_cov @ means[c] + np.log(priors[c])
        scores.append(term + const)
    return np.array(scores).argmax(axis=0)
```

#### Why It Matters

- Interpretable: gives explicit linear boundaries.
- Probabilistic: outputs class posteriors.
- Efficient: closed-form solution (no gradient descent).
- Robust: handles small datasets gracefully.
- Foundational: basis for Fisher's discriminant and QDA.

#### A Gentle Proof (Why It Works)

From Bayes' rule:

$$
P(C_k \mid x) \propto \exp\left(-\frac{1}{2}(x - \mu_k)^\top \Sigma^{-1}(x - \mu_k)\right) P(C_k)
$$

Taking logs and simplifying, quadratic terms in $x$ cancel because $\Sigma$ is shared.
The result is a linear function of $x$:

$$
\delta_k(x) = x^\top \Sigma^{-1} \mu_k - \frac{1}{2} \mu_k^\top \Sigma^{-1} \mu_k + \log P(C_k)
$$

Thus, class boundaries are linear hyperplanes.

#### Try It Yourself

1. Create a 2D dataset with two Gaussian clusters.
2. Fit LDA and plot boundary line.
3. Compare with Logistic Regression, note similarity.
4. Add imbalance, check how priors shift the line.
5. Try $K > 2$ and visualize multi-class regions.

#### Test Cases

| Dataset            | Classes | Expected Boundary               |
| ------------------ | ------- | ------------------------------- |
| Two Gaussian blobs | 2       | Linear line between centroids   |
| Unequal covariance | 2       | LDA struggles (try QDA)         |
| Multi-class        | 3       | Piecewise linear regions        |
| Imbalanced classes | 2       | Boundary shifts toward majority |

#### Complexity

- Training: $O(n d^2)$ (covariance computation)
- Prediction: $O(K d^2)$ per sample
- Space: $O(K d + d^2)$

Linear Discriminant Analysis is where statistics meets geometry, boundaries emerge from likelihoods, separating classes with lines of equal belief.

# Section 92. Ensemble Methods 

### 911. Bagging (Bootstrap Aggregation)

Bagging, short for Bootstrap Aggregation, is an ensemble method that improves the stability and accuracy of machine learning models by combining multiple versions of the same algorithm trained on random subsets of the data. It's particularly effective for high-variance models like decision trees.

#### What Problem Are We Solving?

Many models, especially decision trees, are unstable, small changes in the training data can produce very different results.
Bagging reduces variance by training multiple models on different bootstrap samples and averaging their predictions.

We aim to construct an ensemble predictor:

$$
\hat{f}*{\text{bag}}(x) = \frac{1}{B} \sum*{b=1}^{B} \hat{f}^{(b)}(x)
$$

where each $\hat{f}^{(b)}$ is trained on a different bootstrap sample (random sample with replacement).

#### How Does It Work (Plain Language)?

Bagging is like gathering many opinions.
Each model sees a slightly different dataset (due to random sampling), learns its own perspective, and then the ensemble averages or votes over all predictions.

Step-by-step:

| Step | Action                      | Description                                                       |
| ---- | --------------------------- | ----------------------------------------------------------------- |
| 1    | Sample with Replacement | Create $B$ bootstrap datasets, each the same size as the original |
| 2    | Train Models            | Train base learner (e.g., decision tree) on each sample           |
| 3    | Aggregate Predictions   | For regression: average; for classification: majority vote        |
| 4    | Final Output            | Combined ensemble prediction                                      |

Because each base model sees a slightly different version of the data, their errors partially cancel out when aggregated.

#### Example

If $B = 3$ and the models predict $[0.8, 0.6, 0.9]$ (regression),
the ensemble prediction is:

$$
\hat{y} = \frac{0.8 + 0.6 + 0.9}{3} = 0.7667
$$

For classification, if votes are `[A, A, B]`, majority vote gives `A`.

#### Tiny Code (Easy Versions)

Python (Bagging with Decision Trees)

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier

def bagging_predict(X_train, y_train, X_test, B=10):
    n = len(X_train)
    preds = []
    for _ in range(B):
        idx = np.random.choice(n, n, replace=True)
        Xb, yb = X_train[idx], y_train[idx]
        model = DecisionTreeClassifier()
        model.fit(Xb, yb)
        preds.append(model.predict(X_test))
    preds = np.array(preds)
    # Majority vote across models
    y_pred = [np.bincount(col).argmax() for col in preds.T]
    return np.array(y_pred)
```

C (Outline)

```c
// For each bootstrap iteration:
// 1. Sample training data with replacement
// 2. Train base model (e.g. decision tree)
// 3. Store predictions
// 4. Combine via majority vote or average
```

#### Why It Matters

- Reduces variance: stabilizes noisy learners.
- Improves generalization: especially for decision trees.
- Parallelizable: models are trained independently.
- Foundation for Random Forests: which add random feature selection.

Bagging is especially useful when:

- The base model has high variance (like decision trees)
- There's enough data to create diverse bootstrap samples

#### A Gentle Proof (Why It Works)

Let $\hat{f}(x)$ be a base learner with variance $\text{Var}[\hat{f}(x)]$ and covariance $\rho$ between learners.
The ensemble variance is:

$$
\text{Var}[\hat{f}_{\text{bag}}(x)] = \rho , \text{Var}[\hat{f}(x)] + \frac{1 - \rho}{B} , \text{Var}[\hat{f}(x)]
$$

As $B$ grows, the second term shrinks, reducing total variance.
If base models are uncorrelated ($\rho \approx 0$), bagging greatly improves stability.

#### Try It Yourself

1. Train a single decision tree, note its accuracy.
2. Train a bagged ensemble of 20 trees, compare stability.
3. Plot decision boundaries, bagging smooths jagged edges.
4. Try on noisy datasets, variance reduction is evident.
5. Compare with Random Forest (adds feature randomness).

#### Test Cases

| Dataset       | Base Learner  | Behavior                     |
| ------------- | ------------- | ---------------------------- |
| Noisy data    | Decision Tree | Variance reduced             |
| Smooth data   | Linear Model  | Little improvement           |
| Large dataset | Any           | Ensemble converges           |
| Small dataset | Tree          | Bootstrapping adds diversity |

#### Complexity

- Training: $O(B \times T)$, where $T$ is cost of base learner
- Prediction: $O(B)$ per sample
- Space: $O(B)$ models

Bagging is the wisdom of the crowd, many unstable learners combining their voices to produce one strong, stable prediction.

### 912. Random Forest

A Random Forest is an ensemble of decision trees, each trained on a different random subset of data and features. By combining bagging with feature randomness, it builds a collection of diverse trees whose collective vote reduces overfitting and improves generalization.

#### What Problem Are We Solving?

Even with bagging, if each decision tree sees the same features, they might make similar splits, leading to correlated errors.
Random Forests fix this by adding feature randomness, ensuring each tree explores different dimensions of the feature space.

Given:

- $B$ trees
- $m$ randomly selected features at each split (typically $m = \sqrt{d}$ for classification)

The final ensemble prediction is:

For classification:
$$
\hat{y} = \arg\max_c \sum_{b=1}^{B} \mathbf{1}!\big(\hat{y}^{(b)} = c\big)
$$

For regression:
$$
\hat{y} = \frac{1}{B} \sum_{b=1}^{B} \hat{y}^{(b)}
$$

#### How Does It Work (Plain Language)?

Think of a forest as a team of decision trees, each one grows on a slightly different dataset and focuses on different features.
Individually, trees may overfit, but together, they generalize well.

Step-by-step:

| Step | Action                    | Description                                           |
| ---- | ------------------------- | ----------------------------------------------------- |
| 1    | Bootstrap Sampling    | Draw random samples (with replacement) for each tree  |
| 2    | Feature Subsampling   | At each node, randomly pick $m$ features              |
| 3    | Train Trees           | Grow full decision trees without pruning              |
| 4    | Aggregate Predictions | Majority vote or average over trees                   |
| 5    | Out-of-Bag Estimate   | Use unused samples for validation (no extra test set) |

#### Example

Suppose you train $B = 5$ trees, each using a different bootstrap sample and random subset of features.
Their predictions for a new input are `[A, B, A, A, B]`.
The final majority vote is:

$$
\hat{y} = A
$$

#### Tiny Code (Easy Versions)

Python (Simplified Random Forest)

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier

def random_forest_predict(X_train, y_train, X_test, B=10, m=None):
    n, d = X_train.shape
    if m is None:
        m = int(np.sqrt(d))
    preds = []
    for _ in range(B):
        # Bootstrap sample
        idx = np.random.choice(n, n, replace=True)
        Xb, yb = X_train[idx], y_train[idx]
        # Train tree with feature subsampling
        features = np.random.choice(d, m, replace=False)
        model = DecisionTreeClassifier(max_features=m)
        model.fit(Xb[:, features], yb)
        preds.append(model.predict(X_test[:, features]))
    preds = np.array(preds)
    y_pred = [np.bincount(col).argmax() for col in preds.T]
    return np.array(y_pred)
```

C (Outline)

```c
// For each tree:
// 1. Sample data with replacement
// 2. Randomly select subset of features at each split
// 3. Train decision tree
// 4. Aggregate predictions via vote or average
```

#### Why It Matters

- Reduces variance: decorrelated trees = more stable ensemble
- Handles high dimensions: feature subsampling improves scalability
- Built-in validation: out-of-bag (OOB) score estimates test error
- Robust: works with both categorical and numerical features
- Default strong baseline: great performance with minimal tuning

#### A Gentle Proof (Why It Works)

Variance of a bagged model:
$$
\text{Var}\big[\hat{f}_{\text{bag}}\big] = \rho , \sigma^2 + \frac{1 - \rho}{B} \sigma^2
$$

Adding feature randomness reduces $\rho$ (correlation) between trees, which lowers ensemble variance even further.
Thus, Random Forests outperform plain bagging when features are correlated.

#### Try It Yourself

1. Train a Random Forest on a noisy dataset.
2. Compare its accuracy with a single Decision Tree.
3. Inspect `feature_importances_`, which features matter most?
4. Adjust number of trees $B$ and features $m$.
5. Evaluate out-of-bag (OOB) error vs. test set error.

#### Test Cases

| Dataset             | Base Learner  | Behavior                            |
| ------------------- | ------------- | ----------------------------------- |
| Noisy data          | Decision Tree | Lower variance                      |
| Correlated features | Decision Tree | Feature subsampling helps           |
| High-dimensional    | Decision Tree | Random feature selection essential  |
| Small dataset       | Decision Tree | Bootstrap diversity adds robustness |

#### Complexity

- Training: $O(B \times n \log n)$ (per tree)
- Prediction: $O(B \times \text{depth})$ per sample
- Space: $O(B)$ trees stored

A Random Forest is a crowd of decision trees, each seeing the world differently, together, they make a balanced, well-grounded judgment.

### 913. AdaBoost (Adaptive Boosting)

AdaBoost, short for Adaptive Boosting, is a powerful ensemble method that builds a strong classifier from a collection of weak learners, often shallow decision stumps. Each learner focuses more on the mistakes of the previous ones, "adapting" as the ensemble grows.

#### What Problem Are We Solving?

Many simple models (like one-level decision trees) perform only slightly better than random guessing.
AdaBoost amplifies their strength by combining them into a weighted ensemble that focuses on hard-to-classify examples.

We aim to build a final classifier:

$$
F(x) = \sum_{t=1}^{T} \alpha_t , h_t(x)
$$

where:

- $h_t(x)$ is the weak learner at iteration $t$
- $\alpha_t$ is its weight (how much trust we give it)

The final prediction is:

$$
\hat{y} = \text{sign}\big(F(x)\big)
$$

#### How Does It Work (Plain Language)?

AdaBoost is like a teacher who keeps re-teaching the hardest questions.
After each round, it increases the weight of misclassified points so that the next learner pays more attention to them.

Step-by-step:

| Step | Action                     | Description                                                          |
| ---- | -------------------------- | -------------------------------------------------------------------- |
| 1    | Initialize             | All samples get equal weight $w_i = \frac{1}{n}$                     |
| 2    | Train Weak Learner     | Fit $h_t(x)$ to weighted data                                        |
| 3    | Compute Error          | $\varepsilon_t = \sum_i w_i , \mathbf{1}(h_t(x_i) \ne y_i)$          |
| 4    | Compute Learner Weight | $\alpha_t = \frac{1}{2} \ln \frac{1 - \varepsilon_t}{\varepsilon_t}$ |
| 5    | Update Weights         | $w_i \leftarrow w_i , e^{-\alpha_t y_i h_t(x_i)}$                    |
| 6    | Normalize              | Scale $w_i$ so $\sum_i w_i = 1$                                      |
| 7    | Repeat                 | For $t = 1, 2, \dots, T$                                             |

Hard examples get higher weights, so subsequent learners focus on them.

#### Example

Suppose a weak learner gets 80% accuracy ($\varepsilon = 0.2$).
Its weight is:

$$
\alpha = \frac{1}{2} \ln\frac{1 - 0.2}{0.2} = 0.693
$$

Misclassified points get their weights increased by $e^{+\alpha}$, making them more important next round.

#### Tiny Code (Easy Versions)

Python (Binary AdaBoost with Decision Stumps)

```python
import numpy as np

def adaboost(X, y, T=10):
    n = len(y)
    w = np.ones(n) / n
    models, alphas = [], []

    for _ in range(T):
        # Train weak learner (simple threshold)
        thresh = np.random.choice(X[:, 0])
        preds = np.where(X[:, 0] < thresh, 1, -1)
        err = np.sum(w * (preds != y)) / np.sum(w)

        if err > 0.5: 
            continue

        alpha = 0.5 * np.log((1 - err) / (err + 1e-9))
        w *= np.exp(-alpha * y * preds)
        w /= np.sum(w)

        models.append(thresh)
        alphas.append(alpha)

    def predict(X_test):
        total = np.zeros(len(X_test))
        for thresh, alpha in zip(models, alphas):
            preds = np.where(X_test[:, 0] < thresh, 1, -1)
            total += alpha * preds
        return np.sign(total)

    return predict
```

C (Outline)

```c
// Initialize sample weights equally
// For each round:
//   1. Train weak learner on weighted data
//   2. Compute error and alpha
//   3. Update sample weights
// Combine all weak learners with weighted vote
```

#### Why It Matters

- Boosts weak models into strong ones.
- Focuses on hard cases, adaptive weighting.
- Theoretical guarantee: minimizes exponential loss.
- No overfitting for small T (if weak learners are simple).
- Foundation for gradient boosting methods.

Common base learners:

- Decision stumps (1-split trees)
- Small depth decision trees
- Simple linear classifiers

#### A Gentle Proof (Why It Works)

AdaBoost minimizes the exponential loss:

$$
L = \sum_{i=1}^{n} e^{-y_i F(x_i)}
$$

Each iteration chooses $h_t$ and $\alpha_t$ to greedily reduce $L$.
As $F(x)$ grows, correctly classified points ($y_i F(x_i) > 0$) get tiny weights, while errors dominate the next round's objective.
Thus, the ensemble naturally focuses on mistakes and builds margin over time.

#### Try It Yourself

1. Train AdaBoost with $T = 10$ decision stumps.
2. Plot sample weights, see focus shift to errors.
3. Increase $T$: bias decreases, variance stabilizes.
4. Compare with Bagging, AdaBoost is sequential, not parallel.
5. Check performance on noisy data, too large $T$ may overfit.

#### Test Cases

| Dataset                 | Base Learner | Behavior                    |
| ----------------------- | ------------ | --------------------------- |
| Simple linear separable | Stumps       | Boosts to perfect accuracy  |
| Overlapping classes     | Stumps       | Focuses on ambiguous points |
| High noise              | Stumps       | Overfits if $T$ too large   |
| Balanced dataset        | Stumps       | Fast convergence            |

#### Complexity

- Training: $O(T \cdot n \cdot C)$ (weak learner cost $C$)
- Prediction: $O(T)$ per sample
- Space: $O(T)$ weak models

AdaBoost is an iterative amplifier, it listens to every mistake, learns from it, and builds a chorus of weak voices into one strong, confident decision.

### 914. Gradient Boosting

Gradient Boosting is a powerful ensemble technique that builds models stage by stage, each new learner corrects the residual errors of the previous ensemble by following the gradient of a loss function. It generalizes AdaBoost to arbitrary differentiable losses and base learners.

#### What Problem Are We Solving?

We want a way to combine many weak learners (like shallow trees) into a strong one by minimizing a loss function:

$$
L = \sum_{i=1}^n \ell(y_i, F(x_i))
$$

where:

- $F(x)$ is the ensemble model, built as a sum of weak learners
- $\ell(y, \hat{y})$ is a differentiable loss (e.g., squared error, log-loss)

We iteratively add learners that move $F(x)$ in the direction of the negative gradient of the loss with respect to predictions.

#### How Does It Work (Plain Language)?

Think of it like gradient descent in function space.
Each weak learner $h_t(x)$ learns to predict residuals, the direction to reduce error, not the labels directly.
We then update the model by adding a scaled version of that learner.

Step-by-step (for regression):

| Step | Action                             | Description                                                                   |
| ---- | ---------------------------------- | ----------------------------------------------------------------------------- |
| 1    | Initialize Model               | $F_0(x) = \arg\min_c \sum_i \ell(y_i, c)$                                     |
| 2    | For each iteration $t = 1..T$: |                                                                               |
| 2a   | Compute pseudo-residuals       | $r_i^{(t)} = -\frac{\partial \ell(y_i, F_{t-1}(x_i))}{\partial F_{t-1}(x_i)}$ |
| 2b   | Fit weak learner                   | $h_t(x)$ to $(x_i, r_i^{(t)})$                                                |
| 2c   | Compute step size                  | $\gamma_t = \arg\min_\gamma \sum_i \ell(y_i, F_{t-1}(x_i) + \gamma h_t(x_i))$ |
| 2d   | Update model                       | $F_t(x) = F_{t-1}(x) + \nu \gamma_t h_t(x)$                                   |
| 3    | Output final model             | $F_T(x)$                                                                      |

The learning rate $\nu$ ($0 < \nu \le 1$) controls how much each learner contributes.

#### Example (Squared Error)

For $\ell(y, F(x)) = \frac{1}{2}(y - F(x))^2$,

$$
r_i^{(t)} = y_i - F_{t-1}(x_i)
$$

So each new learner fits residuals (errors) directly.

#### Tiny Code (Easy Versions)

Python (Simplified Gradient Boosting for Regression)

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor

def gradient_boosting(X, y, T=10, lr=0.1, depth=1):
    n = len(y)
    F = np.mean(y) * np.ones(n)
    trees = []
    for _ in range(T):
        residuals = y - F
        tree = DecisionTreeRegressor(max_depth=depth)
        tree.fit(X, residuals)
        F += lr * tree.predict(X)
        trees.append(tree)
    return F, trees
```

C (Outline)

```c
// Initialize model prediction to mean(y)
// For each iteration:
//   1. Compute residuals = y - prediction
//   2. Fit weak learner to residuals
//   3. Update prediction += learning_rate * new_prediction
```

#### Why It Matters

- General framework: supports many losses (regression, classification).
- Flexible: any differentiable loss function.
- Accurate: successive correction yields high precision.
- Controllable: learning rate + tree depth balance bias/variance.
- Foundation: for modern implementations like XGBoost, LightGBM, CatBoost.

#### A Gentle Proof (Why It Works)

We build $F(x)$ incrementally to minimize:

$$
L(F) = \sum_i \ell(y_i, F(x_i))
$$

Each step adds $h_t(x)$ that approximates the negative gradient:

$$
r_i^{(t)} = -\frac{\partial L}{\partial F(x_i)} = -\frac{\partial \ell(y_i, F(x_i))}{\partial F(x_i)}
$$

Thus, the update:

$$
F_t(x) = F_{t-1}(x) + \nu \gamma_t h_t(x)
$$

acts like gradient descent, reducing loss at each iteration.

#### Try It Yourself

1. Fit Gradient Boosting with $T=10$, $lr=0.1$ on simple regression data.
2. Plot predictions as learners accumulate, watch fit improve.
3. Compare with AdaBoost, note difference in update logic.
4. Experiment with deeper trees (reduce bias).
5. Decrease learning rate (increase $T$), smoother convergence.

#### Test Cases

| Dataset           | Loss          | Expected Behavior        |
| ----------------- | ------------- | ------------------------ |
| Linear regression | Squared error | Fits linearly            |
| Nonlinear pattern | Squared error | Approximates curve       |
| Classification    | Log-loss      | Logistic boosting        |
| High noise        | Squared error | Overfits if $T$ too high |

#### Complexity

- Training: $O(T \cdot n \cdot C)$ (weak learner cost $C$)
- Prediction: $O(T)$ per sample
- Space: $O(T)$ learners

Gradient Boosting is boosting with direction, each step follows the gradient, improving not by reweighting mistakes, but by *learning the shape of the loss itself*.

### 915. XGBoost (Extreme Gradient Boosting)

XGBoost (Extreme Gradient Boosting) is a high-performance, regularized implementation of gradient boosting. It extends the basic framework with second-order optimization, shrinkage, column sampling, and built-in regularization, achieving both speed and accuracy on large-scale datasets.

#### What Problem Are We Solving?

While Gradient Boosting provides strong accuracy, it can be slow and prone to overfitting.
XGBoost addresses these by:

1. Adding regularization (L1 and L2) to penalize complexity.
2. Using second-order gradients for more accurate updates.
3. Employing column subsampling and shrinkage to improve generalization.
4. Implementing optimized tree building for large datasets.

We still build a model:

$$
F(x) = \sum_{t=1}^T f_t(x), \quad f_t \in \mathcal{F}
$$

where each $f_t$ is a regression tree, and $\mathcal{F}$ is the space of trees.

#### Objective Function

The training objective at step $t$ is:

$$
\mathcal{L}^{(t)} = \sum_{i=1}^n \ell(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t)
$$

Using a second-order Taylor expansion, we approximate:

$$
\mathcal{L}^{(t)} \approx \sum_{i=1}^n \Big[ g_i f_t(x_i) + \tfrac{1}{2} h_i f_t(x_i)^2 \Big] + \Omega(f_t)
$$

where:

- $g_i = \frac{\partial \ell(y_i, \hat{y}_i^{(t-1)})}{\partial \hat{y}_i^{(t-1)}}$
- $h_i = \frac{\partial^2 \ell(y_i, \hat{y}_i^{(t-1)})}{\partial \hat{y}_i^{(t-1)2}}$

and the regularization term is:

$$
\Omega(f_t) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2
$$

where:

- $\gamma$: penalty for number of leaves
- $\lambda$: L2 regularization on leaf weights $w_j$

#### How Does It Work (Plain Language)?

At each iteration, XGBoost fits a new tree $f_t$ to the gradient and curvature of the loss.
Each leaf has a weight computed analytically to minimize the approximate loss.
This second-order information allows more precise optimization than simple residual fitting.

Step-by-step:

| Step | Action                   | Description                                                                                                                                         |
| ---- | ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | Initialize           | $\hat{y}^{(0)} = \arg\min_c \sum_i \ell(y_i, c)$                                                                                                    |
| 2    | Compute Gradients    | $g_i, h_i$ for each training point                                                                                                                  |
| 3    | Fit Tree             | Use $g_i, h_i$ to find best splits and leaf weights                                                                                                 |
| 4    | Compute Leaf Weights | $w_j^* = - \frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}$                                                                             |
| 5    | Compute Split Gain   | $\text{Gain} = \frac{1}{2}\Big[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}\Big] - \gamma$ |
| 6    | Add Tree             | $\hat{y}^{(t)} = \hat{y}^{(t-1)} + \eta f_t(x)$                                                                                                     |
| 7    | Repeat               | Until $T$ trees or early stopping                                                                                                                   |

#### Tiny Code (Conceptual Skeleton)

Python (Conceptual Gradient Step)

```python
# Pseudo-code (not actual implementation)
for t in range(T):
    g = grad(loss, y_pred)
    h = hess(loss, y_pred)
    tree = build_tree(X, g, h, lambda_, gamma)
    y_pred += eta * tree.predict(X)
```

XGBoost's C++ backend builds trees efficiently using histograms and parallelization.

#### Why It Matters

- Regularized: avoids overfitting via $\lambda$ and $\gamma$.
- Second-order optimization: faster, more accurate steps.
- Shrinkage (learning rate): gradual updates improve generalization.
- Column subsampling: decorrelates trees, reduces variance.
- Highly optimized: supports parallel and distributed training.

XGBoost is widely used in Kaggle competitions, finance, and industrial ML systems due to its balance of speed, accuracy, and robustness.

#### A Gentle Proof (Why It Works)

The objective is approximated as:

$$
\tilde{\mathcal{L}}^{(t)} = \sum_{j=1}^T \Big[ G_j w_j + \tfrac{1}{2}(H_j + \lambda)w_j^2 \Big] + \gamma T
$$

Minimizing with respect to $w_j$ gives:

$$
w_j^* = -\frac{G_j}{H_j + \lambda}
$$

Plugging back in gives optimal split gain:

$$
\text{Gain} = \frac{1}{2}\left(\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}\right) - \gamma
$$

Each split maximizes gain, reducing loss greedily with regularization.

#### Try It Yourself

1. Train XGBoost on a small dataset (binary classification).
2. Inspect feature importances, which features dominate?
3. Adjust $\eta$ (learning rate): small $\eta$ + large $T$ = smoother convergence.
4. Tune $\lambda$, $\gamma$ to balance bias and variance.
5. Compare with plain Gradient Boosting, faster convergence, less overfitting.

#### Test Cases

| Dataset               | Task          | Behavior                        |
| --------------------- | ------------- | ------------------------------- |
| Binary classification | Log-loss      | Fast convergence                |
| Noisy regression      | MSE           | Stable via regularization       |
| Wide data             | Many features | Column subsampling helps        |
| Overfitting-prone     | Any           | Controlled by $\lambda, \gamma$ |

#### Complexity

- Training: $O(T \cdot n \log n)$ (optimized tree splits)
- Prediction: $O(T \cdot \text{depth})$ per sample
- Space: $O(T)$ trees

XGBoost is Gradient Boosting, evolved, combining second-order learning, regularization, and computational engineering into a fast, reliable powerhouse for structured data.

### 916. LightGBM (Light Gradient Boosting Machine)

LightGBM is a highly efficient implementation of gradient boosting that focuses on speed, scalability, and memory efficiency. It introduces innovative techniques like histogram-based learning, leaf-wise growth, and Gradient-based One-Side Sampling (GOSS) to handle massive datasets without sacrificing accuracy.

#### What Problem Are We Solving?

Traditional gradient boosting (and even XGBoost) can become slow and memory-intensive on large, high-dimensional data.
LightGBM tackles this by:

1. Reducing computation with histogram-based splits.
2. Growing trees leaf-wise, not level-wise (deeper, more accurate trees).
3. Using sampling techniques to focus on impactful data points.
4. Supporting categorical features natively.

Its goal: fast training, low memory, high accuracy, especially for large-scale structured data.

#### Core Ideas

LightGBM builds additive models:

$$
F(x) = \sum_{t=1}^T f_t(x)
$$

Each $f_t(x)$ is a decision tree trained to minimize a second-order loss approximation:

$$
\mathcal{L}^{(t)} \approx \sum_{i=1}^{n} \Big[g_i f_t(x_i) + \tfrac{1}{2} h_i f_t(x_i)^2\Big] + \Omega(f_t)
$$

but with histogram-based splits and optimized sampling.

#### How Does It Work (Plain Language)?

LightGBM streamlines tree boosting through three innovations:

1. Histogram-based Splitting:
   Continuous features are bucketed into discrete bins.
   This drastically reduces split search cost from $O(n \cdot d)$ to $O(B \cdot d)$,
   where $B$ is the number of bins (e.g. 255).

2. Leaf-wise Tree Growth (Best-First):
   Instead of expanding all leaves evenly (level-wise), LightGBM picks the leaf with the largest loss reduction to grow next.
   This allows deeper, more focused trees.

3. GOSS (Gradient-based One-Side Sampling):
   Keeps samples with large gradients (important for loss reduction)
   and randomly drops some with small gradients, reducing data while preserving learning direction.

#### Step-by-step Summary

| Step | Action                | Description                                 |
| ---- | --------------------- | ------------------------------------------- |
| 1    | Compute Gradients | $g_i, h_i$ for all samples                  |
| 2    | Bucket Features   | Build histograms of gradients by bin        |
| 3    | Split Search      | For each feature, evaluate gain per bin     |
| 4    | Select Split      | Maximize loss reduction                     |
| 5    | Leaf-wise Growth  | Expand leaf with largest gain               |
| 6    | Apply GOSS        | Keep high-gradient samples, sample low ones |
| 7    | Repeat            | Until max leaves or stopping criterion      |

#### Example: Split Gain Formula

The gain from splitting leaf $j$ is computed as:

$$
\text{Gain} = \frac{1}{2}\left( \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right) - \gamma
$$

where $G$ and $H$ are the sums of gradients and Hessians in left/right child leaves.

#### Tiny Code (Conceptual Sketch)

Python (Conceptual)

```python
# Conceptual illustration, not full implementation
for t in range(T):
    g, h = compute_gradients(y, y_pred)
    hist = build_histograms(X, g, h, bins=255)
    best_split = find_best_leafwise_split(hist)
    tree = grow_tree(best_split)
    y_pred += lr * tree.predict(X)
```

In Practice: use `lightgbm.LGBMClassifier` or `LGBMRegressor` from the library.

#### Why It Matters

- Speed: Histogram binning + GOSS = faster than XGBoost.
- Accuracy: Leaf-wise trees focus on high-gain splits.
- Memory efficiency: Uses compressed bins, not raw floats.
- Scalable: Handles millions of rows and features.
- Native support: Categorical variables and GPU acceleration.

LightGBM is especially effective on large tabular datasets common in finance, recommender systems, and competitions.

#### A Gentle Proof (Why It Works)

The histogram trick approximates continuous features into bins $B_j$:

$$
\sum_{i \in B_j} g_i, \quad \sum_{i \in B_j} h_i
$$

Splits are evaluated using these aggregated values, preserving gradient statistics while reducing cost.

For GOSS:

- Keep top $a%$ high-gradient samples.
- Sample $b%$ of low-gradient samples.
- Apply a correction factor $\frac{1-a}{b}$ to maintain unbiasedness.

This maintains a valid estimate of total gradient direction, ensuring convergence to the same optimum as full data.

#### Try It Yourself

1. Train LightGBM on a dataset with 1M samples, note speed.
2. Compare with XGBoost, see 2–3× faster training.
3. Tune `num_leaves`: higher → lower bias, higher variance.
4. Adjust `max_bin` for accuracy/speed tradeoff.
5. Enable `categorical_feature`, no one-hot encoding needed.

#### Test Cases

| Dataset           | Task           | Behavior                        |
| ----------------- | -------------- | ------------------------------- |
| Large tabular     | Classification | Fast, high accuracy             |
| Small dataset     | Regression     | Similar to XGBoost              |
| Categorical-heavy | Classification | Built-in handling               |
| Noisy data        | Any            | May overfit, tune `num_leaves` |

#### Complexity

- Training: $O(T \cdot B \cdot d)$ (with $B$ bins, $d$ features)
- Prediction: $O(T \cdot \text{depth})$
- Space: $O(B \cdot d)$ for histograms

LightGBM is gradient boosting on turbo mode, fast, memory-efficient, and laser-focused, designed for modern data scale and speed.

### 917. CatBoost (Categorical Boosting)

CatBoost is a gradient boosting library designed specifically to handle categorical features efficiently and avoid prediction shift (target leakage). It uses ordered boosting and target encoding with permutation-based statistics, producing fast, accurate, and stable models, especially for tabular data.

#### What Problem Are We Solving?

Many gradient boosting libraries require manual encoding (like one-hot or label encoding) for categorical features, which can lead to:

- Inefficient memory use (especially for high-cardinality features)
- Overfitting due to target leakage (when target information leaks into training)

CatBoost solves this by:

1. Natively encoding categorical features using permutation-driven target statistics.
2. Using ordered boosting, which prevents data from "seeing the future" during training.

The model still follows the standard boosting formula:

$$
F(x) = \sum_{t=1}^T f_t(x)
$$

but with special handling for categorical variables and ordered learning.

#### How Does It Work (Plain Language)?

CatBoost builds trees like Gradient Boosting, but adds two key innovations:

1. Ordered Target Encoding (OTEs):
   Instead of using the global target mean for encoding a category (which leaks information),
   CatBoost computes progressive averages over random permutations of the dataset.
   Each sample's encoding uses only past samples.

   Example for category $c$:
   $$
   \text{Enc}(x_i) = \frac{\sum_{j < i, x_j = c} y_j + a \cdot P}{N_{<i, c} + a}
   $$
   where:

   * $N_{<i, c}$ = count of category $c$ before position $i$
   * $P$ = prior (global mean)
   * $a$ = smoothing parameter

2. Ordered Boosting:
   Instead of fitting residuals on the same dataset, it simulates online learning:
   each iteration uses only data that would have been available at that point.
   This avoids target leakage and improves generalization.

#### Step-by-step Summary

| Step | Action                     | Description                                         |
| ---- | -------------------------- | --------------------------------------------------- |
| 1    | Random Permutation     | Shuffle training data                               |
| 2    | Encode Categories      | Compute target statistics in order                  |
| 3    | Compute Gradients      | Based on current predictions                        |
| 4    | Fit Weak Learner       | Train tree on transformed data                      |
| 5    | Apply Ordered Boosting | Use only past information                           |
| 6    | Add Tree               | Update ensemble $F_t(x) = F_{t-1}(x) + \eta f_t(x)$ |

#### Example: Target Encoding

Suppose we have a categorical feature "Color" with samples in order:

| Sample | Color | Target | Encoding (step-by-step)     |
| ------ | ----- | ------ | --------------------------- |
| 1      | Red   | 1      | $(a \cdot P) / a$           |
| 2      | Blue  | 0      | $(a \cdot P) / a$           |
| 3      | Red   | 0      | $(a \cdot P + 1) / (a + 1)$ |
| 4      | Blue  | 1      | $(a \cdot P + 0) / (a + 1)$ |

Each sample only uses past target values for encoding, avoiding leakage.

#### Tiny Code (Easy Versions)

Python (with CatBoost library)

```python
from catboost import CatBoostClassifier

model = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    cat_features=['color', 'brand', 'city'],
    loss_function='Logloss',
    verbose=0
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

C (Outline)

```c
// Pseudocode:
// 1. Shuffle data randomly
// 2. For each categorical feature, compute ordered target encoding
// 3. Train gradient boosting trees on encoded data using ordered boosting
```

#### Why It Matters

- No preprocessing required: handles categorical features natively.
- Avoids target leakage: uses ordered statistics and boosting.
- Fast and accurate: efficient C++ core with CPU/GPU support.
- Great for tabular data: especially with mixed feature types.
- Robust: less tuning required than XGBoost or LightGBM.

CatBoost shines when datasets include categorical variables, small to medium size, and non-linear interactions.

#### A Gentle Proof (Why It Works)

CatBoost combats target leakage by making feature transformations causally consistent:

- For encoding, only earlier samples influence the current one.
- For boosting, each model sees predictions computed without access to the same sample.

This ensures unbiased gradient estimation, improving stability and generalization.

#### Try It Yourself

1. Train CatBoost on a dataset with categorical columns.
2. Compare with one-hot encoding + XGBoost.
3. Observe better accuracy and less overfitting.
4. Visualize feature importances, categories are meaningful.
5. Try `model.get_feature_importance(prettified=True)` for insights.

#### Test Cases

| Dataset           | Feature Type | Behavior                    |
| ----------------- | ------------ | --------------------------- |
| Categorical-heavy | Many strings | Superior to one-hot methods |
| Continuous-only   | Numeric      | Similar to XGBoost          |
| High-cardinality  | IDs or names | Efficient via OTE           |
| Small dataset     | Mixed        | Stable, low overfitting     |

#### Complexity

- Training: $O(T \cdot n \log n)$ (tree building)
- Prediction: $O(T \cdot \text{depth})$ per sample
- Space: $O(T)$ trees + encoded features

CatBoost is boosting done right for categorical data, blending probabilistic encoding, ordered learning, and gradient optimization into one cohesive system.

### 918. Stacking (Stacked Generalization)

Stacking (or Stacked Generalization) is an ensemble learning technique that combines multiple base models by training a meta-model to learn how to best blend their predictions. Instead of simple averaging or voting, it learns the optimal combination directly from data.

#### What Problem Are We Solving?

Individual models capture different patterns or biases in data.
Simple ensembling (like bagging or boosting) may not exploit these complementary strengths effectively.

Stacking learns a meta-level model to optimally weight or combine the outputs of base learners, often yielding higher accuracy and robust generalization.

We aim to build a model:

$$
\hat{y} = g(f_1(x), f_2(x), \ldots, f_m(x))
$$

where:

- $f_i(x)$ = base learner predictions
- $g(\cdot)$ = meta-model (e.g. linear regression, logistic regression)

#### How Does It Work (Plain Language)?

Stacking is a two-level learning system:

1. Level-0 (Base Learners): Train multiple diverse models (trees, SVMs, neural nets, etc.).
2. Level-1 (Meta Learner): Train a model on the out-of-fold predictions of base learners to predict the final output.

By using out-of-fold (OOF) predictions, we ensure that the meta-model is trained on unseen data, preventing information leakage.

#### Step-by-step Summary

| Step | Action                      | Description                                                 |
| ---- | --------------------------- | ----------------------------------------------------------- |
| 1    | Split Data into Folds   | Create $K$ folds for cross-validation                       |
| 2    | Train Base Models       | Each $f_i$ trained on $(K-1)$ folds, predicts held-out fold |
| 3    | Collect OOF Predictions | Build new dataset of base predictions                       |
| 4    | Train Meta-Model        | Fit $g$ on OOF predictions vs. true labels                  |
| 5    | Final Model             | Base learners on full data + meta-model on full OOF         |

The final prediction for a new $x$:

$$
\hat{y} = g(f_1(x), f_2(x), \ldots, f_m(x))
$$

#### Example

Suppose you have 3 base models: Logistic Regression ($f_1$), Random Forest ($f_2$), and SVM ($f_3$).
You generate their predictions on validation folds and stack them as new features:

| Sample | $f_1(x)$ | $f_2(x)$ | $f_3(x)$ | $y$ |
| ------ | -------- | -------- | -------- | --- |
| 1      | 0.2      | 0.4      | 0.3      | 0   |
| 2      | 0.8      | 0.9      | 0.7      | 1   |
| 3      | 0.6      | 0.5      | 0.4      | 1   |

Then train a meta-model $g$ (like Logistic Regression) on this table.

#### Tiny Code (Easy Versions)

Python (Stacking Example)

```python
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

def stacking_train_predict(X, y, X_test, folds=5):
    base_models = [
        LogisticRegression(max_iter=1000),
        RandomForestClassifier(n_estimators=100),
        GradientBoostingClassifier(),
        SVC(probability=True)
    ]
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    meta_features = np.zeros((len(y), len(base_models)))
    
    # Level-0: generate out-of-fold predictions
    for j, model in enumerate(base_models):
        oof_pred = np.zeros(len(y))
        for train_idx, val_idx in kf.split(X):
            model.fit(X[train_idx], y[train_idx])
            oof_pred[val_idx] = model.predict_proba(X[val_idx])[:, 1]
        meta_features[:, j] = oof_pred
    
    # Level-1: train meta-model
    meta_model = LogisticRegression()
    meta_model.fit(meta_features, y)
    
    # Predict
    test_meta = np.column_stack([m.fit(X, y).predict_proba(X_test)[:, 1] for m in base_models])
    return meta_model.predict(test_meta)
```

C (Outline)

```c
// 1. Train each base model on K-1 folds, predict held-out fold
// 2. Combine out-of-fold predictions into meta dataset
// 3. Train meta-model on this new dataset
// 4. Use trained base + meta models for final predictions
```

#### Why It Matters

- Combines strengths of diverse models.
- Learns combination weights instead of assuming equal importance.
- Reduces generalization error by leveraging complementary patterns.
- Flexible framework: any model can serve as base or meta learner.

Stacking is often the final layer in competitive ML pipelines, blending tree-based, linear, and deep models.

#### A Gentle Proof (Why It Works)

By training $g$ on OOF predictions, stacking approximates the conditional expectation of $y$ given model outputs:

$$
g^*(f_1(x), \dots, f_m(x)) = \mathbb{E}[y \mid f_1(x), \dots, f_m(x)]
$$

This ensures unbiased learning since $g$ only sees predictions generated without using the true fold during fitting, avoiding overfitting and improving generalization.

#### Try It Yourself

1. Choose 3–5 base learners (diverse architectures).
2. Use 5-fold OOF stacking to create meta-features.
3. Train a simple meta-model (Logistic or Ridge).
4. Compare with averaging or voting, stacking often wins.
5. Try 2-level stacking (stack of stacks) for advanced setups.

#### Test Cases

| Dataset                | Base Models        | Meta-Model        | Behavior           |
| ---------------------- | ------------------ | ----------------- | ------------------ |
| Mixed linear/nonlinear | Logistic, RF, GBM  | Logistic          | Best of both       |
| Noisy                  | Trees, SVM         | Ridge             | Smooth combination |
| Small dataset          | Simple models      | Low depth         | Avoid overfitting  |
| Large dataset          | Many base learners | Strong meta-model | Gains accuracy     |

#### Complexity

- Training: $O(K \cdot M \cdot C)$ (folds × base models × cost)
- Prediction: $O(M + 1)$ per sample
- Space: $O(M)$ models + meta dataset

Stacking is where models learn to collaborate, a meta-learner orchestrates their voices, turning a chorus of predictions into harmony.

### 919. Voting Classifier

A Voting Classifier is one of the simplest ensemble methods, it combines predictions from multiple models and decides the final output by majority vote (for classification) or average (for regression). It doesn't learn combination weights; instead, it relies on the collective "wisdom" of its models.

#### What Problem Are We Solving?

Individual models can be unstable or biased.
Averaging their opinions often reduces variance and improves robustness.

The voting classifier offers a quick, interpretable way to aggregate multiple models without extra training.

We aim to combine $M$ models:

For classification:
$$
\hat{y} = \arg\max_{c} \sum_{m=1}^{M} \mathbf{1}(\hat{y}^{(m)} = c)
$$

For regression:
$$
\hat{y} = \frac{1}{M} \sum_{m=1}^{M} \hat{y}^{(m)}
$$

#### How Does It Work (Plain Language)?

Imagine consulting several experts. Each gives a prediction.
You don't judge who's better, you simply take the majority opinion (hard voting) or average their confidence (soft voting).

Two main modes:

1. Hard Voting: Use predicted class labels; majority wins.
2. Soft Voting: Use predicted probabilities; take the class with highest average probability.

#### Step-by-step Summary

| Step | Action              | Description                                                                  |
| ---- | ------------------- | ---------------------------------------------------------------------------- |
| 1    | Train Models    | Fit each base model independently on the same dataset                        |
| 2    | Get Predictions | For each model, compute $\hat{y}^{(m)}$ (hard) or $p^{(m)}(y \mid x)$ (soft) |
| 3    | Combine         | Aggregate via majority (hard) or mean (soft)                                 |
| 4    | Decide Output   | Return the class with most votes or highest mean probability                 |

#### Example

Suppose 3 classifiers give predictions:

| Model               | Prediction |
| ------------------- | ---------- |
| Logistic Regression | A          |
| Decision Tree       | A          |
| SVM                 | B          |

Hard Voting:

- Votes: A(2), B(1) → Final: A

Soft Voting (probabilities):

| Model    | P(A) | P(B) |
| -------- | ---- | ---- |
| Logistic | 0.7  | 0.3  |
| Tree     | 0.6  | 0.4  |
| SVM      | 0.4  | 0.6  |

Averaged:

- $P(A) = 0.57$, $P(B) = 0.43$ → Final: A

#### Tiny Code (Easy Versions)

Python (Hard & Soft Voting Example)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

clf1 = LogisticRegression(max_iter=1000)
clf2 = DecisionTreeClassifier()
clf3 = SVC(probability=True)

voting_clf = VotingClassifier(
    estimators=[('lr', clf1), ('dt', clf2), ('svm', clf3)],
    voting='soft'  # 'hard' for majority vote
)

voting_clf.fit(X_train, y_train)
y_pred = voting_clf.predict(X_test)
```

C (Outline)

```c
// 1. Train multiple base models
// 2. Collect predictions
// 3. Hard vote: pick class with majority votes
// 4. Soft vote: average predicted probabilities, pick highest
```

#### Why It Matters

- Simple & effective: improves accuracy without extra complexity.
- Low risk: doesn't require tuning or meta-learning.
- Stable: smooths out individual model errors.
- Versatile: works with any classifiers or regressors.

Ideal when:

- Models are independent or have diverse biases.
- You want quick ensemble gains without cross-validation or meta-models.

#### A Gentle Proof (Why It Works)

If each model has error rate $p < 0.5$ and errors are independent,
the probability that the majority vote is wrong decreases exponentially with the number of models:

$$
P(\text{majority wrong}) = \sum_{k = \lceil M/2 \rceil}^{M} \binom{M}{k} p^k (1-p)^{M-k}
$$

This is the Condorcet Jury Theorem: the majority decision is more likely correct than any individual voter, assuming independence and competence.

#### Try It Yourself

1. Combine logistic regression, decision tree, and kNN.
2. Try both `voting='hard'` and `voting='soft'`.
3. Compare accuracy with individual models.
4. Add a poor-performing model, see how it dilutes performance.
5. Test on noisy data, ensembles resist overfitting.

#### Test Cases

| Dataset              | Mode   | Behavior           |
| -------------------- | ------ | ------------------ |
| Balanced classes     | Hard   | Stable accuracy    |
| Probabilistic models | Soft   | Better calibration |
| Correlated models    | Either | Limited gain       |
| Diverse models       | Soft   | Strong improvement |

#### Complexity

- Training: $O(\sum_m C_m)$ (sum of individual costs)
- Prediction: $O(M)$ per sample
- Space: $O(M)$ models

The Voting Classifier is the simplest ensemble, no blending, no boosting, just a democratic vote where every model has a say, and consensus drives accuracy.

### 920. Snapshot Ensemble

A Snapshot Ensemble is an elegant technique that turns a single training run into multiple models by capturing the network's weights at different points during training, typically when it converges to different local minima using a cyclical learning rate schedule. These snapshots are later combined to form an ensemble, improving generalization without extra training cost.

#### What Problem Are We Solving?

Ensembles improve accuracy by averaging predictions from diverse models, but training many deep networks is expensive.
Snapshot Ensembles solve this by reusing one training trajectory, capturing multiple diverse states of the same model as it oscillates through different local minima.

We want to approximate an ensemble:

$$
\hat{y} = \frac{1}{M} \sum_{m=1}^{M} f(x; \theta_m)
$$

where each $\theta_m$ is a snapshot of the model parameters at a different point in training.

#### How Does It Work (Plain Language)?

Instead of training $M$ models independently, we vary the learning rate cyclically, letting the optimizer move into and out of multiple minima.
At the end of each cycle (when the learning rate is small), we save the weights.
Each saved model is then used in the final ensemble.

Step-by-step:

| Step | Action                         | Description                                         |
| ---- | ------------------------------ | --------------------------------------------------- |
| 1    | Initialize Model           | Start from random weights                           |
| 2    | Use Cyclical Learning Rate | Schedule (like cosine annealing) repeats $M$ cycles |
| 3    | At Each Cycle End          | Save weights (snapshot)                             |
| 4    | Continue Training          | Restart learning rate high again                    |
| 5    | After Training             | Combine all snapshots (average predictions)         |

Thus, $M$ snapshots $\Rightarrow$ $M$ models, all from one training process.

#### Learning Rate Schedule (Cosine Annealing)

The learning rate $\eta(t)$ is decayed and restarted periodically:

$$
\eta(t) = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min}) \left( 1 + \cos\left( \frac{\pi \bmod(t-1, T/M)}{T/M} \right) \right)
$$

where:

- $T$ = total iterations
- $M$ = number of cycles
- $\eta_{\max}, \eta_{\min}$ = bounds of learning rate

This cyclical schedule ensures exploration of multiple basins of attraction.

#### Example

Suppose you plan $M = 3$ snapshots in 90 epochs:

- Each cycle = 30 epochs
- Learning rate restarts every 30 epochs
- Save model at epoch 30, 60, 90

Final prediction:

$$
\hat{y} = \frac{1}{3}(f_1(x) + f_2(x) + f_3(x))
$$

#### Tiny Code (Easy Versions)

Python (PyTorch-style Example)

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

model = MyNetwork()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = CosineAnnealingLR(optimizer, T_max=30)
snapshots = []

for epoch in range(90):
    train_one_epoch(model, optimizer, data_loader)
    scheduler.step()
    if (epoch + 1) % 30 == 0:
        snapshots.append(model.state_dict().copy())

# Ensemble predictions
def ensemble_predict(x):
    preds = []
    for s in snapshots:
        model.load_state_dict(s)
        preds.append(model(x))
    return sum(preds) / len(preds)
```

C (Outline)

```c
// Train model with cosine-annealed learning rate
// Every cycle:
//   1. Save model weights (snapshot)
//   2. Restart learning rate high
// After training: average predictions across snapshots
```

#### Why It Matters

- Free ensemble: No additional training runs.
- Diversity from a single trajectory: Each snapshot represents a different local minimum.
- Improved generalization: Averages out variance and noise.
- Elegant and efficient: Same training time as one model.

It's particularly useful in deep learning where full ensembles are costly.

#### A Gentle Proof (Why It Works)

In non-convex optimization, stochastic gradient descent may converge to multiple local minima depending on the learning rate.
By restarting the learning rate, we push the model out of one basin and into another, capturing multiple diverse hypotheses.
Averaging their outputs reduces generalization error:

$$
E[(f(x) - y)^2] = \text{bias}^2 + \text{variance}
$$

Snapshot averaging lowers the variance term.

#### Try It Yourself

1. Train a CNN with cosine annealing and $M = 3$ cycles.
2. Save weights at end of each cycle.
3. Compare accuracy:

   * Single final model
   * Snapshot ensemble (average predictions)
4. Observe improved validation accuracy and smoother learning.
5. Increase $M$ for more diversity (up to diminishing returns).

#### Test Cases

| Dataset         | Model  | Behavior                                |
| --------------- | ------ | --------------------------------------- |
| CIFAR-10        | CNN    | Higher test accuracy                    |
| MNIST           | MLP    | More stable predictions                 |
| Large network   | ResNet | Similar to multi-model ensemble         |
| Limited compute | Any    | Great tradeoff: ensemble at single cost |

#### Complexity

- Training: $O(T)$ (same as single model)
- Prediction: $O(M)$ per sample (ensemble averaging)
- Space: $O(M)$ snapshots

Snapshot Ensembles give you the best of both worlds, the power of ensembling and the efficiency of single-model training, harnessing learning rate cycles to explore and capture diverse solutions.

# Section 93. Gradient Methods 

### 921. Gradient Descent

Gradient Descent is the foundational optimization algorithm for training machine learning models. It works by iteratively moving parameters in the direction that reduces the loss function, following the slope downhill until reaching a (local) minimum.

#### What Problem Are We Solving?

Most learning problems involve minimizing a loss function
$$
L(\theta)
$$
over parameters $\theta$. Closed-form solutions are rare for nonlinear functions, so we need an iterative method to approach the minimum.

Gradient Descent updates parameters by moving opposite to the gradient of the loss:

$$
\theta \leftarrow \theta - \eta , \nabla_\theta L(\theta)
$$

where:

- $\nabla_\theta L(\theta)$ is the gradient (direction of steepest ascent),
- $\eta$ is the learning rate, controlling step size.

The intuition:
If the gradient points uphill, moving in the opposite direction reduces the loss.

#### How Does It Work (Plain Language)?

Imagine standing on a curved surface representing your loss function.
You want to find the lowest valley.
At each step:

1. Measure the slope (gradient).
2. Step downhill a small amount.
3. Repeat until the slope flattens (near minimum).

Step-by-step:

| Step | Action                    | Description                                                  |
| ---- | ------------------------- | ------------------------------------------------------------ |
| 1    | Initialize parameters | Choose random $\theta_0$                                     |
| 2    | Compute gradient      | $\nabla_\theta L(\theta_t)$                                  |
| 3    | Update                | $\theta_{t+1} = \theta_t - \eta , \nabla_\theta L(\theta_t)$ |
| 4    | Check convergence     | Stop if gradient small or change minimal                     |
| 5    | Repeat                | Until loss stabilizes                                        |

#### Example

Suppose
$$
L(\theta) = \theta^2
$$
Then
$$
\nabla_\theta L = 2\theta
$$
With $\eta = 0.1$ and $\theta_0 = 1.0$:

| Iteration | $\theta$ | $L(\theta)$ |
| --------- | -------- | ----------- |
| 0         | 1.000    | 1.000       |
| 1         | 0.800    | 0.640       |
| 2         | 0.640    | 0.410       |
| 3         | 0.512    | 0.262       |

Each step moves $\theta$ closer to 0 (the minimum).

#### Tiny Code (Easy Versions)

Python

```python
import numpy as np

def gradient_descent(L_grad, theta0, lr=0.1, steps=100):
    theta = theta0
    for _ in range(steps):
        grad = L_grad(theta)
        theta -= lr * grad
    return theta

# Example: minimize L(theta) = theta^2
L_grad = lambda t: 2 * t
theta_opt = gradient_descent(L_grad, theta0=1.0, lr=0.1)
print(theta_opt)
```

C (Outline)

```c
double theta = 1.0, lr = 0.1;
for (int i = 0; i < 100; i++) {
    double grad = 2 * theta;  // d/dθ (θ²)
    theta -= lr * grad;
}
printf("Optimal θ: %f\n", theta);
```

#### Why It Matters

- Universal optimizer: works for any differentiable loss.
- Foundation of deep learning: every neural network is trained via gradient descent (or variants).
- Conceptual bridge: introduces learning rate, convergence, and curvature intuition.
- Extensible: leads to SGD, Momentum, Adam, etc.

#### A Gentle Proof (Why It Works)

Near the minimum, Taylor expansion:

$$
L(\theta + \Delta \theta) \approx L(\theta) + \nabla_\theta L(\theta)^\top \Delta \theta
$$

To reduce $L$, choose $\Delta \theta$ in direction $-\nabla_\theta L$.
If $\eta$ is small enough, each update decreases loss monotonically.

Convergence rate (for convex $L$):

$$
L(\theta_t) - L(\theta^*) \le \frac{C}{t}
$$

where $C$ depends on smoothness of $L$.

#### Try It Yourself

1. Minimize $L(\theta) = (\theta - 3)^2$.
2. Experiment with $\eta = 0.01, 0.1, 1.0$.
3. Plot trajectory of $\theta_t$ over iterations.
4. Try multivariate $L(\theta_1, \theta_2) = \theta_1^2 + 2\theta_2^2$.
5. Visualize gradient arrows on contour plot.

#### Test Cases

| Loss       | Initial $\theta$ | $\eta$ | Convergence            |
| ---------- | ---------------- | ------ | ---------------------- |
| $\theta^2$ | 1.0              | 0.1    | Smooth descent         |
| $\theta^2$ | 1.0              | 1.0    | Overshoot oscillations |
| $\theta^2$ | 1.0              | 0.01   | Slow convergence       |

#### Complexity

- Per iteration: $O(d)$ (for $d$ parameters)
- Memory: $O(d)$
- Convergence: depends on curvature and step size

Gradient Descent is the engine of learning, a simple, steady march downhill, powering everything from linear regression to deep neural networks.

### 922. Stochastic Gradient Descent (SGD)

Stochastic Gradient Descent (SGD) is a faster, noisier cousin of classical gradient descent.
Instead of using all training samples to compute the gradient at each step, SGD updates parameters using a single sample (or small batch), giving it speed, randomness, and the ability to escape shallow minima.

#### What Problem Are We Solving?

In large-scale learning, computing the full gradient

$$
\nabla_\theta L(\theta) = \frac{1}{N} \sum_{i=1}^{N} \nabla_\theta \ell(x_i, y_i; \theta)
$$

is too slow, especially when $N$ is huge.

SGD approximates the full gradient using a single random sample (or mini-batch):

$$
\theta \leftarrow \theta - \eta , \nabla_\theta \ell(x_i, y_i; \theta)
$$

This gives a cheap, noisy estimate that works well on average.

#### How Does It Work (Plain Language)?

Think of classical gradient descent as checking every data point before taking a step, very precise, but slow.
SGD, instead, takes a quick glance at one example, makes a guess, adjusts a bit, and keeps going, learning through many small, noisy updates.

Over time, these noisy steps average out to move roughly in the right direction.

Step-by-step:

| Step | Action                | Description                                  |
| ---- | --------------------- | -------------------------------------------- |
| 1    | Shuffle Data      | Randomize order of training samples          |
| 2    | Pick One Sample   | $(x_i, y_i)$                                 |
| 3    | Compute Gradient  | $g_i = \nabla_\theta \ell(x_i, y_i; \theta)$ |
| 4    | Update Parameters | $\theta \gets \theta - \eta g_i$             |
| 5    | Repeat            | For all samples (1 epoch), then reshuffle    |

#### Example

Suppose loss is
$$
L(\theta) = \frac{1}{N}\sum_{i=1}^{N} (\theta - x_i)^2
$$

Full gradient:
$$
\nabla_\theta L = 2(\theta - \bar{x})
$$

SGD update with one sample $x_i$:
$$
\theta \gets \theta - \eta \cdot 2(\theta - x_i)
$$

Each update pulls $\theta$ a little toward a single data point, gradually converging toward $\bar{x}$.

#### Tiny Code (Easy Versions)

Python

```python
import numpy as np

def sgd(X, y, lr=0.1, epochs=10):
    n = len(y)
    theta = 0.0
    for _ in range(epochs):
        for i in np.random.permutation(n):
            grad = 2 * (theta - X[i])
            theta -= lr * grad
    return theta
```

C (Outline)

```c
for (int epoch = 0; epoch < epochs; epoch++) {
    shuffle(data);
    for (int i = 0; i < n; i++) {
        double grad = 2 * (theta - x[i]);
        theta -= lr * grad;
    }
}
```

#### Why It Matters

- Fast and scalable: one update per sample.
- Online learning: can adapt continuously as new data arrives.
- Noise helps escape local minima.
- Foundation for deep learning: modern optimizers (Adam, RMSProp) are built on SGD.

It's the optimizer that lets neural networks train on massive datasets without waiting forever.

#### A Gentle Proof (Why It Works)

If $\mathbb{E}[\nabla_\theta \ell(x_i, y_i; \theta)] = \nabla_\theta L(\theta)$,
then on average, SGD follows the true gradient:

$$
\mathbb{E}[\theta_{t+1}] = \theta_t - \eta , \nabla_\theta L(\theta_t)
$$

So even though each step is noisy, the expected direction is correct.
With a decaying learning rate $\eta_t$, SGD converges to the global minimum for convex functions.

#### Try It Yourself

1. Implement SGD for linear regression.
2. Compare convergence vs full-batch gradient descent.
3. Plot the noisy trajectory over a contour map of the loss.
4. Try different learning rates: $\eta = 0.01$, $0.1$, $1.0$.
5. Add a mini-batch size (e.g. 16 or 32) and observe smoothing.

#### Test Cases

| Dataset    | Batch Size | Behavior                        |
| ---------- | ---------- | ------------------------------- |
| Small      | 1          | High variance, quick updates    |
| Large      | 32         | Smooth path, stable convergence |
| Non-convex | 1          | Escapes local minima            |
| Streaming  | 1          | Online adaptation               |

#### Complexity

- Per iteration: $O(1)$
- Per epoch: $O(N)$
- Memory: $O(d)$
- Convergence: depends on learning rate schedule and noise level

Stochastic Gradient Descent trades precision for speed, learning by trial, noise, and many small steps, making it the beating heart of modern optimization.

### 923. Mini-Batch Stochastic Gradient Descent (Mini-Batch SGD)

Mini-Batch SGD strikes a balance between full-batch gradient descent and purely stochastic updates.
Instead of using all samples or just one, it updates parameters using small batches of data, combining efficiency, stability, and speed, making it the default choice in deep learning.

#### What Problem Are We Solving?

Full-batch gradient descent:

$$
\nabla_\theta L(\theta) = \frac{1}{N} \sum_{i=1}^{N} \nabla_\theta \ell(x_i, y_i; \theta)
$$

is accurate but slow.

Stochastic gradient descent:

$$
\nabla_\theta L(\theta) \approx \nabla_\theta \ell(x_i, y_i; \theta)
$$

is fast but noisy.

Mini-batch SGD finds the middle ground:

$$
\nabla_\theta L(\theta) \approx \frac{1}{B} \sum_{i=1}^{B} \nabla_\theta \ell(x_i, y_i; \theta)
$$

where $B$ is the batch size (e.g. 16, 32, 64).
This approach uses parallel computation and smoother gradients.

#### How Does It Work (Plain Language)?

Instead of looking at one data point (SGD) or all (full-batch), we take a handful, a mini-batch, to approximate the direction downhill.
It's like taking several opinions before deciding, rather than polling everyone or just one.

Step-by-step:

| Step | Action                 | Description                                     |
| ---- | ---------------------- | ----------------------------------------------- |
| 1    | Shuffle dataset    | Randomize order                                 |
| 2    | Split into batches | Each of size $B$                                |
| 3    | Loop over batches  | Compute gradient per batch                      |
| 4    | Update parameters  | $\theta \gets \theta - \eta , g_{\text{batch}}$ |
| 5    | Repeat per epoch   | Continue until convergence                      |

Each batch gives a gradient estimate with lower variance than SGD, and much faster than full-batch.

#### Example

Suppose $N = 1000$ samples, batch size $B = 50$:
Each epoch = $1000 / 50 = 20$ updates.
Each update:

$$
g = \frac{1}{50} \sum_{i=1}^{50} \nabla_\theta \ell(x_i, y_i)
$$

and
$$
\theta \gets \theta - \eta g
$$

The model sees all data per epoch, but in chunks, efficient for vectorized computation (e.g. GPUs).

#### Tiny Code (Easy Versions)

Python

```python
import numpy as np

def minibatch_sgd(X, y, lr=0.1, batch_size=32, epochs=10):
    n = len(y)
    theta = 0.0
    for _ in range(epochs):
        indices = np.random.permutation(n)
        for start in range(0, n, batch_size):
            end = start + batch_size
            batch = indices[start:end]
            grad = np.mean(2 * (theta - X[batch]))
            theta -= lr * grad
    return theta
```

C (Outline)

```c
for (int epoch = 0; epoch < epochs; epoch++) {
    shuffle(data);
    for (int i = 0; i < n; i += batch_size) {
        int end = min(i + batch_size, n);
        double grad = 0;
        for (int j = i; j < end; j++)
            grad += 2 * (theta - x[j]);
        grad /= (end - i);
        theta -= lr * grad;
    }
}
```

#### Why It Matters

- Computationally efficient: leverages vectorization and parallelism.
- Less noisy: gradient estimate is more accurate than single-sample SGD.
- Faster convergence: smoother path to minimum.
- GPU-friendly: batches fit hardware constraints.

It's the standard optimizer for training neural networks.

#### A Gentle Proof (Why It Works)

If $\mathbb{E}[g_B] = \nabla_\theta L(\theta)$,
then batch gradient $g_B$ is an unbiased estimator of the true gradient.

Variance decreases as batch size $B$ increases:

$$
\text{Var}(g_B) = \frac{\sigma^2}{B}
$$

Hence, larger batches give smoother but costlier updates.
Mini-batches find a sweet spot between cost and stability.

#### Try It Yourself

1. Train linear regression with batch sizes 1, 32, 128.
2. Plot loss vs iteration, see smoother curves for larger batches.
3. Observe compute time per epoch.
4. Use batch size = 32 (common default).
5. Experiment with decaying learning rate $\eta_t$.

#### Test Cases

| Batch Size     | Behavior                           |
| -------------- | ---------------------------------- |
| 1              | High noise, fast updates           |
| 32             | Balanced performance               |
| 512            | Smooth path, more compute per step |
| N (full batch) | Exact gradient, slow               |

#### Complexity

- Per iteration: $O(B \cdot d)$
- Per epoch: $O(N \cdot d)$
- Memory: $O(B \cdot d)$
- Convergence: faster in practice due to variance reduction

Mini-Batch SGD is the workhorse of modern optimization, efficient, smooth, and perfectly tuned for parallel computation.

### 924. Momentum

Momentum is a powerful enhancement to gradient descent that helps it move faster in the right direction and smooth out oscillations. It does this by adding a velocity term that remembers past updates, so the optimizer can accelerate down long slopes and resist noisy zigzags.

#### What Problem Are We Solving?

Plain gradient descent can get stuck or slow down when the surface curves differently along different directions, like a narrow valley.

It may oscillate across steep walls, inching forward slowly:

$$
\theta \leftarrow \theta - \eta , \nabla_\theta L(\theta)
$$

Momentum solves this by accumulating past gradients in a velocity term, allowing consistent movement in the dominant direction.

#### The Update Rule

Let $v_t$ be the velocity at step $t$. Then:

$$
v_t = \beta v_{t-1} + (1 - \beta) \nabla_\theta L(\theta_t)
$$

$$
\theta_{t+1} = \theta_t - \eta v_t
$$

where:

- $\beta$ (0.8–0.99) controls how much of the past momentum to keep,
- $\eta$ is the learning rate.

This is equivalent to applying an exponential moving average to gradients.

#### How Does It Work (Plain Language)?

Imagine rolling a ball down a hill.
Without momentum, it stops at every bump.
With momentum, it keeps rolling, carrying energy from previous steps, smoothing small oscillations and speeding up descent.

Step-by-step:

| Step | Action                           | Description                             |
| ---- | -------------------------------- | --------------------------------------- |
| 1    | Initialize $\theta_0$, $v_0 = 0$ | Start parameters and velocity           |
| 2    | Compute gradient $g_t$           | $g_t = \nabla_\theta L(\theta_t)$       |
| 3    | Update velocity                  | $v_t = \beta v_{t-1} + (1 - \beta) g_t$ |
| 4    | Update parameters                | $\theta_{t+1} = \theta_t - \eta v_t$    |
| 5    | Repeat                           | Until convergence                       |

#### Example

Suppose we're minimizing
$$
L(\theta_1, \theta_2) = 100\theta_1^2 + \theta_2^2
$$

The surface is steep along $\theta_1$, flat along $\theta_2$.
Without momentum: steps oscillate along $\theta_1$.
With momentum: velocity accumulates along $\theta_2$, reducing zigzags and converging faster.

#### Tiny Code (Easy Versions)

Python

```python
import numpy as np

def gradient_descent_momentum(L_grad, theta0, lr=0.1, beta=0.9, steps=100):
    theta = theta0
    v = 0
    for _ in range(steps):
        grad = L_grad(theta)
        v = beta * v + (1 - beta) * grad
        theta -= lr * v
    return theta

# Example: minimize L(theta) = theta^2
L_grad = lambda t: 2 * t
theta_opt = gradient_descent_momentum(L_grad, theta0=1.0)
print(theta_opt)
```

C (Outline)

```c
double theta = 1.0, v = 0.0, lr = 0.1, beta = 0.9;
for (int i = 0; i < 100; i++) {
    double grad = 2 * theta;
    v = beta * v + (1 - beta) * grad;
    theta -= lr * v;
}
printf("Optimal θ: %f\n", theta);
```

#### Why It Matters

- Accelerates convergence on flat regions.
- Reduces oscillation in steep valleys.
- Simple and robust, works with any gradient-based optimizer.
- Foundation for advanced methods (Nesterov, Adam, RMSProp).

Momentum turns gradient descent from a cautious walker into a rolling ball, smooth, decisive, and fast.

#### A Gentle Proof (Why It Works)

Momentum approximates a first-order low-pass filter over gradients:

$$
v_t \approx \sum_{k=0}^{t} (1 - \beta)\beta^{t-k} \nabla_\theta L(\theta_k)
$$

So the update direction becomes a weighted average of past gradients.
This averaging suppresses high-frequency noise, stabilizing convergence.

#### Try It Yourself

1. Train gradient descent on $L(\theta_1, \theta_2) = 100\theta_1^2 + \theta_2^2$.
2. Compare with and without momentum.
3. Try $\beta = 0.8, 0.9, 0.99$, higher $\beta$ = smoother motion.
4. Plot trajectories on contour maps.
5. Experiment with large learning rate $\eta$, momentum allows bigger steps.

#### Test Cases

| Surface        | $\beta$ | Behavior            |
| -------------- | ------- | ------------------- |
| Flat           | 0.9     | Faster acceleration |
| Steep valley   | 0.9     | Less oscillation    |
| Noisy gradient | 0.99    | Smoother updates    |
| Convex         | 0.8     | Converges steadily  |

#### Complexity

- Per iteration: $O(d)$
- Memory: $O(d)$ (for velocity)
- Convergence: faster than vanilla GD on ill-conditioned surfaces

Momentum remembers where it's been, letting the optimizer glide past bumps and speed through valleys toward the minimum.

### 925. Nesterov Accelerated Gradient (NAG)

Nesterov Accelerated Gradient (NAG) refines standard momentum by adding a "lookahead" step, it anticipates where the parameters are moving and computes the gradient after taking that step.
This small change provides stronger convergence guarantees and better control near minima.

#### What Problem Are We Solving?

Regular momentum helps gradient descent gain speed, but it can overshoot when nearing the optimum.
Nesterov fixes this by peeking ahead before applying the gradient, ensuring updates are guided by where the parameters are *going*, not where they *were*.

Momentum update:
$$
v_t = \beta v_{t-1} + (1 - \beta)\nabla_\theta L(\theta_t)
$$

Nesterov update:
$$
v_t = \beta v_{t-1} + (1 - \beta)\nabla_\theta L(\theta_t - \eta \beta v_{t-1})
$$

Then:
$$
\theta_{t+1} = \theta_t - \eta v_t
$$

The gradient is evaluated at the lookahead point, giving earlier correction.

#### How Does It Work (Plain Language)?

Think of running downhill:

- Momentum: you move based on your current velocity, then adjust after seeing where you end up.
- Nesterov: you look ahead before stepping, adjusting your direction proactively.

This anticipation helps prevent overshooting and makes convergence smoother.

Step-by-step:

| Step | Action                      | Description                                            |
| ---- | --------------------------- | ------------------------------------------------------ |
| 1    | Compute lookahead point | $\theta_{\text{look}} = \theta_t - \eta \beta v_{t-1}$ |
| 2    | Compute gradient        | $g_t = \nabla_\theta L(\theta_{\text{look}})$          |
| 3    | Update velocity         | $v_t = \beta v_{t-1} + (1 - \beta) g_t$                |
| 4    | Update parameters       | $\theta_{t+1} = \theta_t - \eta v_t$                   |

#### Example

Suppose $L(\theta) = \theta^2$.
We start with $\theta_0 = 1.0$, $\beta = 0.9$, $\eta = 0.1$.

Each step:

1. Look ahead: $\theta_{\text{look}} = \theta - 0.1 \times 0.9 v$
2. Compute gradient at $\theta_{\text{look}}$
3. Update $v$, then $\theta$

This "peek" allows faster, more stable convergence than standard momentum.

#### Tiny Code (Easy Versions)

Python

```python
import numpy as np

def nesterov(L_grad, theta0, lr=0.1, beta=0.9, steps=100):
    theta = theta0
    v = 0.0
    for _ in range(steps):
        lookahead = theta - lr * beta * v
        grad = L_grad(lookahead)
        v = beta * v + (1 - beta) * grad
        theta -= lr * v
    return theta

# Example: minimize L(theta) = theta^2
L_grad = lambda t: 2 * t
theta_opt = nesterov(L_grad, theta0=1.0)
print(theta_opt)
```

C (Outline)

```c
double theta = 1.0, v = 0.0, lr = 0.1, beta = 0.9;
for (int i = 0; i < 100; i++) {
    double lookahead = theta - lr * beta * v;
    double grad = 2 * lookahead;
    v = beta * v + (1 - beta) * grad;
    theta -= lr * v;
}
printf("Optimal θ: %f\n", theta);
```

#### Why It Matters

- Faster convergence on convex problems.
- More stable near minima, corrects direction before stepping.
- Widely used in deep learning (e.g., SGD with Nesterov).
- Theoretically strong: provably optimal for convex optimization.

In practice, NAG improves both speed and stability, often outperforming plain momentum.

#### A Gentle Proof (Why It Works)

Nesterov's method comes from convex optimization theory.
For smooth convex functions:

$$
L(\theta_t) - L(\theta^*) \le O\left(\frac{1}{t^2}\right)
$$

compared to $O\left(\frac{1}{t}\right)$ for standard gradient descent.
The key is anticipation: evaluating gradients at $\theta_t - \eta \beta v_{t-1}$ leads to earlier correction and tighter bounds.

#### Try It Yourself

1. Train on $L(\theta) = \theta^2$ with momentum vs NAG.
2. Plot convergence curves, NAG is faster and smoother.
3. Test $\beta = 0.8, 0.9, 0.99$.
4. Try non-convex loss (e.g. multi-basin), note improved control.
5. Compare training speed on a small neural net.

#### Test Cases

| Surface          | Optimizer | Behavior             |
| ---------------- | --------- | -------------------- |
| Convex quadratic | NAG       | Faster than momentum |
| Non-convex       | NAG       | More stable          |
| Valley-shaped    | NAG       | Less oscillation     |
| High curvature   | NAG       | Controlled updates   |

#### Complexity

- Per iteration: $O(d)$
- Memory: $O(d)$
- Convergence: $O(1/t^2)$ (convex), faster in practice

Nesterov Accelerated Gradient is momentum with foresight, a runner who glances ahead before stepping, reaching the finish line faster and more gracefully.

### 926. AdaGrad (Adaptive Gradient)

AdaGrad, short for *Adaptive Gradient*, automatically adjusts the learning rate for each parameter based on how frequently it's been updated.
Parameters with frequent large gradients get smaller steps, while rarely updated ones get larger steps, perfect for sparse features or imbalanced data.

#### What Problem Are We Solving?

Standard gradient descent uses a single global learning rate:

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t)
$$

But in real data:

- Some parameters update often (high gradient frequency),
- Others rarely change (low frequency).

A single learning rate can either overshoot some or underfit others.
AdaGrad adapts $\eta$ individually per dimension.

#### The Update Rule

Each parameter $\theta_i$ maintains a cumulative sum of squared gradients:

$$
G_{t,i} = G_{t-1,i} + g_{t,i}^2
$$

Then update:

$$
\theta_{t+1,i} = \theta_{t,i} - \frac{\eta}{\sqrt{G_{t,i}} + \epsilon} , g_{t,i}
$$

where:

- $g_{t,i}$ is the gradient of parameter $i$ at step $t$,
- $\epsilon$ avoids division by zero (e.g. $10^{-8}$),
- $\eta$ is the initial learning rate.

So the effective learning rate shrinks over time:

$$
\eta_{t,i} = \frac{\eta}{\sqrt{G_{t,i}} + \epsilon}
$$

#### How Does It Work (Plain Language)?

Think of AdaGrad as giving each parameter its own step size, inversely proportional to how often it's been updated.

- Frequently changing parameters take smaller steps.
- Rare ones take bigger steps to catch up.

This is especially useful in NLP, recommenders, and sparse vectors (e.g. word embeddings).

Step-by-step:

| Step | Action                     | Description                            |
| ---- | -------------------------- | -------------------------------------- |
| 1    | Initialize $\theta$, $G=0$ | Start parameters and accumulators      |
| 2    | Compute gradient $g$       | $\nabla_\theta L(\theta_t)$            |
| 3    | Update accumulator         | $G \gets G + g \odot g$                |
| 4    | Scale learning rate        | $\eta_t = \eta / \sqrt{G + \epsilon}$  |
| 5    | Update parameters          | $\theta \gets \theta - \eta_t \odot g$ |

#### Example

Suppose we optimize $L(\theta_1, \theta_2)$.
Gradients over time:

| Step | $g_1$ | $g_2$ |
| ---- | ----- | ----- |
| 1    | 1.0   | 0.1   |
| 2    | 1.0   | 0.1   |
| 3    | 1.0   | 0.1   |

Then $G_1$ grows faster than $G_2$:

- $\eta_1$ shrinks more, slowing updates,
- $\eta_2$ stays larger, adapting slower-moving parameter.

#### Tiny Code (Easy Versions)

Python

```python
import numpy as np

def adagrad(L_grad, theta0, lr=0.1, eps=1e-8, steps=100):
    theta = theta0
    G = np.zeros_like(theta)
    for _ in range(steps):
        grad = L_grad(theta)
        G += grad  2
        theta -= lr * grad / (np.sqrt(G) + eps)
    return theta

# Example: minimize L(theta) = theta^2
L_grad = lambda t: 2 * t
theta_opt = adagrad(L_grad, np.array([1.0]))
print(theta_opt)
```

C (Outline)

```c
double theta = 1.0, G = 0.0, lr = 0.1, eps = 1e-8;
for (int i = 0; i < 100; i++) {
    double grad = 2 * theta;
    G += grad * grad;
    theta -= lr * grad / (sqrt(G) + eps);
}
printf("Optimal θ: %f\n", theta);
```

#### Why It Matters

- Adaptive step sizes: no manual tuning per parameter.
- Works well with sparse features.
- Monotonic learning rate decay: stabilizes convergence.
- Foundation for RMSProp and Adam.

AdaGrad was one of the first adaptive optimizers, paving the way for modern deep learning methods.

#### A Gentle Proof (Why It Works)

AdaGrad performs a diagonal preconditioning of the gradient:

$$
\theta_{t+1} = \theta_t - \eta G_t^{-1/2} g_t
$$

Here $G_t^{-1/2}$ rescales each dimension based on historical magnitude.
This gives an effective second-order correction, similar in spirit to Newton's method, but simpler.

#### Try It Yourself

1. Optimize $L(\theta_1, \theta_2) = \theta_1^2 + 100\theta_2^2$.
2. Compare vanilla GD vs AdaGrad.
3. Observe smoother path and adaptive speed.
4. Try on sparse feature vector (many zeros).
5. Track $\eta_t$, see how it decays.

#### Test Cases

| Surface        | Behavior                     |
| -------------- | ---------------------------- |
| Sparse         | Fast convergence             |
| Dense          | Step size decays too quickly |
| Long training  | Stops early                  |
| NLP embeddings | Very effective               |

#### Complexity

- Per iteration: $O(d)$
- Memory: $O(d)$ (accumulator $G$)
- Convergence: smooth but may stagnate (too small $\eta_t$)

AdaGrad learns how to learn, scaling each dimension individually, letting rarely updated features speak louder and frequent ones quiet down.

### 927. RMSProp (Root Mean Square Propagation)

RMSProp improves upon AdaGrad by preventing its learning rate from decaying too quickly.
It keeps an exponentially weighted moving average of squared gradients instead of summing them forever, maintaining adaptability while avoiding stagnation.

#### What Problem Are We Solving?

In AdaGrad, the denominator

$$
\sqrt{G_t} = \sqrt{\sum_{i=1}^t g_i^2}
$$

keeps growing, so the effective learning rate

$$
\eta_t = \frac{\eta}{\sqrt{G_t} + \epsilon}
$$

shrinks too much over time.
Training may stop early, especially in non-convex settings like deep neural networks.

RMSProp fixes this by using a decaying average of squared gradients, balancing adaptivity and persistence.

#### The Update Rule

Maintain an exponential moving average of squared gradients:

$$
E[g^2]*t = \beta E[g^2]*{t-1} + (1 - \beta) g_t^2
$$

Then update parameters:

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t} + \epsilon} , g_t
$$

where:

- $\beta$ is the decay rate (typically 0.9),
- $\epsilon$ (e.g. $10^{-8}$) prevents division by zero,
- $\eta$ is the learning rate (often $10^{-3}$ or $10^{-4}$).

#### How Does It Work (Plain Language)?

RMSProp tracks the recent average magnitude of gradients.
If gradients are large in one direction, it shrinks the learning rate there; if small, it increases it.
Unlike AdaGrad, it forgets old gradients, keeping training adaptive throughout.

Step-by-step:

| Step | Action                          | Description                                         |
| ---- | ------------------------------- | --------------------------------------------------- |
| 1    | Initialize $\theta$, $E[g^2]=0$ | Parameters + running average                        |
| 2    | Compute gradient $g_t$          | $\nabla_\theta L(\theta_t)$                         |
| 3    | Update average                  | $E[g^2]*t = \beta E[g^2]*{t-1} + (1 - \beta) g_t^2$ |
| 4    | Scale learning rate             | $\eta_t = \eta / \sqrt{E[g^2]_t + \epsilon}$        |
| 5    | Update parameters               | $\theta \gets \theta - \eta_t g_t$                  |

#### Example

Suppose gradients oscillate:

- Step 1: $g = 1.0$
- Step 2: $g = 0.2$
- Step 3: $g = 0.5$

Instead of summing ($1.0^2 + 0.2^2 + 0.5^2$),
RMSProp uses:

$$
E[g^2]*t = 0.9 E[g^2]*{t-1} + 0.1 g_t^2
$$

which emphasizes recent gradients, letting learning rates adapt continuously.

#### Tiny Code (Easy Versions)

Python

```python
import numpy as np

def rmsprop(L_grad, theta0, lr=0.01, beta=0.9, eps=1e-8, steps=100):
    theta = theta0
    Eg2 = np.zeros_like(theta)
    for _ in range(steps):
        grad = L_grad(theta)
        Eg2 = beta * Eg2 + (1 - beta) * grad  2
        theta -= lr * grad / (np.sqrt(Eg2) + eps)
    return theta

# Example: minimize L(theta) = theta^2
L_grad = lambda t: 2 * t
theta_opt = rmsprop(L_grad, np.array([1.0]))
print(theta_opt)
```

C (Outline)

```c
double theta = 1.0, Eg2 = 0.0, lr = 0.01, beta = 0.9, eps = 1e-8;
for (int i = 0; i < 100; i++) {
    double grad = 2 * theta;
    Eg2 = beta * Eg2 + (1 - beta) * grad * grad;
    theta -= lr * grad / (sqrt(Eg2) + eps);
}
printf("Optimal θ: %f\n", theta);
```

#### Why It Matters

- Adaptive learning rates across dimensions.
- No premature decay, unlike AdaGrad.
- Stable training for deep networks (especially RNNs).
- Default optimizer in many early deep learning frameworks (e.g. TensorFlow's default before Adam).

It's like a thermostat for learning rates, adjusting constantly based on recent temperature (gradient energy).

#### A Gentle Proof (Why It Works)

RMSProp approximates second-order curvature using moving averages:

$$
\mathbb{E}[g^2]_t \approx \text{diag}(H)
$$

so step size per dimension becomes:

$$
\Delta \theta_i \propto \frac{1}{\sqrt{\mathbb{E}[g_i^2]_t}}
$$

acting as a preconditioner, stabilizing updates along steep or flat directions.

#### Try It Yourself

1. Train $L(\theta) = 100\theta_1^2 + \theta_2^2$.
2. Compare AdaGrad vs RMSProp.
3. Plot learning rates over time, RMSProp stays active longer.
4. Try $\beta = 0.9, 0.99$.
5. Observe smoother convergence on non-stationary gradients.

#### Test Cases

| Surface        | Optimizer | Behavior             |
| -------------- | --------- | -------------------- |
| Steep valley   | RMSProp   | Smooth convergence   |
| Sparse data    | RMSProp   | Adapts per dimension |
| Long training  | RMSProp   | Keeps learning       |
| Non-stationary | RMSProp   | Stable updates       |

#### Complexity

- Per iteration: $O(d)$
- Memory: $O(d)$ (running average)
- Convergence: faster, more stable than AdaGrad in deep nets

RMSProp is AdaGrad with a memory, adaptive, forgetful, and tuned for the dynamic landscapes of deep learning.

### 928. Adam (Adaptive Moment Estimation)

Adam, short for *Adaptive Moment Estimation*, combines the strengths of Momentum and RMSProp.
It keeps track of both the mean (first moment) and variance (second moment) of gradients to provide adaptive learning rates and smooth, stable convergence, making it the most widely used optimizer in deep learning.

#### What Problem Are We Solving?

SGD may oscillate or require careful tuning of the learning rate.
Momentum accelerates convergence but lacks adaptivity.
RMSProp adapts learning rates but ignores direction history.

Adam merges both ideas, maintaining:

- Velocity (first moment): exponential moving average of gradients.
- Scale (second moment): exponential moving average of squared gradients.

Together, these stabilize updates and adapt per-parameter step sizes.

#### The Update Rule

At each step $t$:

1. Compute gradient:
   $$
   g_t = \nabla_\theta L(\theta_t)
   $$

2. Update biased first moment (mean):
   $$
   m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
   $$

3. Update biased second moment (variance):
   $$
   v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
   $$

4. Bias correction:
   $$
   \hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
   $$

5. Parameter update:
   $$
   \theta_{t+1} = \theta_t - \eta , \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
   $$

Typical hyperparameters:

- $\eta = 10^{-3}$
- $\beta_1 = 0.9$
- $\beta_2 = 0.999$
- $\epsilon = 10^{-8}$

#### How Does It Work (Plain Language)?

Adam blends momentum's direction memory with RMSProp's adaptive scaling.

Think of it as a smart autopilot:

- The first moment (like velocity) keeps moving consistently downhill.
- The second moment (like shock absorbers) adjusts step size to terrain roughness.

Together, they make learning fast, smooth, and robust to noise.

Step-by-step:

| Step | Action              | Description                                                            |
| ---- | ------------------- | ---------------------------------------------------------------------- |
| 1    | Compute $g_t$       | Gradient at current step                                               |
| 2    | Update $m_t$, $v_t$ | Moving averages of gradient and square                                 |
| 3    | Correct bias        | Adjust for initialization bias                                         |
| 4    | Compute update      | Scale gradient by $\sqrt{v_t}$                                         |
| 5    | Apply step          | $\theta \gets \theta - \eta \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$ |

#### Example

Let $L(\theta) = \theta^2$.
Starting at $\theta_0 = 1.0$:

- $m_t$ smooths gradient direction,
- $v_t$ scales by recent gradient energy,
- Learning rate adapts per step, larger early, smaller later.

#### Tiny Code (Easy Versions)

Python

```python
import numpy as np

def adam(L_grad, theta0, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, steps=100):
    theta = theta0
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    for t in range(1, steps + 1):
        g = L_grad(theta)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g  2)
        m_hat = m / (1 - beta1  t)
        v_hat = v / (1 - beta2  t)
        theta -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return theta

# Example: minimize L(theta) = theta^2
L_grad = lambda t: 2 * t
theta_opt = adam(L_grad, np.array([1.0]))
print(theta_opt)
```

C (Outline)

```c
double theta = 1.0, m = 0.0, v = 0.0;
double lr = 0.001, beta1 = 0.9, beta2 = 0.999, eps = 1e-8;

for (int t = 1; t <= 100; t++) {
    double g = 2 * theta;
    m = beta1 * m + (1 - beta1) * g;
    v = beta2 * v + (1 - beta2) * g * g;
    double m_hat = m / (1 - pow(beta1, t));
    double v_hat = v / (1 - pow(beta2, t));
    theta -= lr * m_hat / (sqrt(v_hat) + eps);
}
printf("Optimal θ: %f\n", theta);
```

#### Why It Matters

- Adaptive learning rates per parameter.
- Momentum for speed, RMS scaling for stability.
- Little hyperparameter tuning needed.
- Works well in deep networks, especially with noisy gradients.

Adam is the default optimizer for most modern deep learning models, from CNNs to Transformers.

#### A Gentle Proof (Why It Works)

Bias correction ensures unbiased estimates of first and second moments:

$$
\mathbb{E}[m_t] = \nabla_\theta L(\theta_t), \quad \mathbb{E}[v_t] = \mathbb{E}[g_t^2]
$$

Using these, Adam approximates a second-order preconditioner:

$$
\Delta \theta \propto \frac{m_t}{\sqrt{v_t}}
$$

which stabilizes steps in ill-conditioned landscapes, giving faster and safer convergence.

#### Try It Yourself

1. Train a neural net with Adam vs SGD.
2. Observe faster convergence.
3. Try $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\eta = 10^{-3}$.
4. Experiment with decoupled weight decay (AdamW).
5. Plot learning curve vs iterations.

#### Test Cases

| Dataset         | Optimizer | Behavior                   |
| --------------- | --------- | -------------------------- |
| Deep net        | Adam      | Fast, smooth convergence   |
| Sparse features | Adam      | Stable per-parameter steps |
| Noisy gradients | Adam      | Robust updates             |
| Flat minima     | Adam      | Settles gracefully         |

#### Complexity

- Per iteration: $O(d)$
- Memory: $O(2d)$ (for $m$, $v$)
- Convergence: fast, stable, widely applicable

Adam is the "automatic transmission" of optimization, combining speed, control, and adaptability for modern machine learning.

### 929. AdamW (Adam with Decoupled Weight Decay)

AdamW is a refined version of Adam that fixes a subtle but important issue: the way weight decay (L2 regularization) is applied.
In standard Adam, L2 penalty interacts incorrectly with the adaptive learning rate.
AdamW decouples weight decay from the gradient update, yielding more accurate regularization and better generalization.

#### What Problem Are We Solving?

In classical gradient descent, L2 regularization (weight decay) is applied as:

$$
\theta \leftarrow \theta - \eta , (\nabla_\theta L + \lambda \theta)
$$

But in Adam, gradients are rescaled by $\sqrt{v_t}$, so adding $\lambda \theta$ inside the gradient term causes the penalty to scale unevenly per parameter, not true weight decay.

AdamW solves this by decoupling weight decay from gradient scaling.

#### The Update Rule

Same as Adam, but with separate weight decay:

1. Gradient moments:
   $$
   m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
   $$
   $$
   v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
   $$

2. Bias correction:
   $$
   \hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
   $$

3. Parameter update with decoupled decay:
   $$
   \theta_{t+1} = \theta_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)
   $$

where $\lambda$ is the weight decay coefficient.

#### How Does It Work (Plain Language)?

In Adam, L2 regularization is mistakenly scaled by adaptive learning rates.
In AdamW, weight decay is applied directly to parameters, just like in SGD.

This ensures a consistent shrinkage per step, independent of the gradient magnitude.

Think of it as:

- Adam: "rescaled" penalty → irregular regularization.
- AdamW: "pure" decay → consistent regularization strength.

#### Step-by-Step Summary

| Step | Action                 | Description                                                            |
| ---- | ---------------------- | ---------------------------------------------------------------------- |
| 1    | Compute gradient $g_t$ | $\nabla_\theta L(\theta_t)$                                            |
| 2    | Update moments         | $m_t$, $v_t$                                                           |
| 3    | Apply bias correction  | $\hat{m}_t$, $\hat{v}_t$                                               |
| 4    | Update weights         | $\theta \gets \theta - \eta , \hat{m}_t / \sqrt{\hat{v}_t + \epsilon}$ |
| 5    | Apply decay            | $\theta \gets \theta - \eta \lambda \theta$                            |

#### Example

For a parameter $\theta = 1.0$,
let $\eta = 0.01$, $\lambda = 0.1$.

The decay step is simple:

$$
\theta \gets \theta - \eta \lambda \theta = \theta(1 - 0.001)
$$

This decay happens independently of gradient magnitude, giving clean regularization.

#### Tiny Code (Easy Versions)

Python

```python
import numpy as np

def adamw(L_grad, theta0, lr=0.001, beta1=0.9, beta2=0.999, wd=0.01, eps=1e-8, steps=100):
    theta = theta0
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    for t in range(1, steps + 1):
        g = L_grad(theta)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g  2)
        m_hat = m / (1 - beta1  t)
        v_hat = v / (1 - beta2  t)
        theta -= lr * (m_hat / (np.sqrt(v_hat) + eps) + wd * theta)
    return theta

# Example: minimize L(theta) = theta^2
L_grad = lambda t: 2 * t
theta_opt = adamw(L_grad, np.array([1.0]))
print(theta_opt)
```

C (Outline)

```c
double theta = 1.0, m = 0.0, v = 0.0;
double lr = 0.001, beta1 = 0.9, beta2 = 0.999, wd = 0.01, eps = 1e-8;

for (int t = 1; t <= 100; t++) {
    double g = 2 * theta;
    m = beta1 * m + (1 - beta1) * g;
    v = beta2 * v + (1 - beta2) * g * g;
    double m_hat = m / (1 - pow(beta1, t));
    double v_hat = v / (1 - pow(beta2, t));
    theta -= lr * (m_hat / (sqrt(v_hat) + eps) + wd * theta);
}
printf("Optimal θ: %f\n", theta);
```

#### Why It Matters

- True weight decay: consistent L2 penalty.
- Better generalization: especially in deep nets.
- Fixes Adam's bias toward larger weights.
- Default in PyTorch (`torch.optim.AdamW`) and Transformers.

AdamW is a must-have for training large neural networks, like BERT and GPT.

#### A Gentle Proof (Why It Works)

In Adam, L2 penalty gets scaled by $\frac{1}{\sqrt{v_t}}$, which breaks theoretical equivalence to weight decay.

By decoupling:

$$
\text{Update: } \theta \gets \theta - \eta \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}
$$

$$
\text{Decay: } \theta \gets (1 - \eta \lambda)\theta
$$

the regularization term acts linearly and consistently, independent of gradient statistics.

#### Try It Yourself

1. Train a small CNN with Adam vs AdamW.
2. Compare validation loss and weight norms.
3. Try $\lambda = 0.01, 0.1, 0.001$.
4. Observe smoother convergence and better test performance.
5. Track norm of $\theta$, AdamW keeps it controlled.

#### Test Cases

| Model       | Optimizer | Behavior                 |
| ----------- | --------- | ------------------------ |
| MLP         | Adam      | Overfits (weights grow)  |
| MLP         | AdamW     | Better generalization    |
| Transformer | AdamW     | Stable training          |
| CNN         | AdamW     | Controlled weight growth |

#### Complexity

- Per iteration: $O(d)$
- Memory: $O(2d)$
- Convergence: same as Adam, but better generalization

AdamW keeps Adam's adaptive brilliance but restores proper regularization, a simple fix with profound impact on modern deep learning.

### 930. L-BFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno)

L-BFGS is a powerful quasi-Newton optimization method that approximates second-order curvature (the Hessian) using only a small amount of memory.
It achieves faster convergence than first-order methods like SGD by modeling the local shape of the loss, without explicitly computing the Hessian.

#### What Problem Are We Solving?

In optimization, Newton's Method updates parameters using the inverse Hessian:

$$
\theta_{t+1} = \theta_t - H_t^{-1} \nabla_\theta L(\theta_t)
$$

But for high-dimensional problems, storing and inverting $H_t$ (size $d \times d$) is too costly, $O(d^2)$ memory and $O(d^3)$ compute.

L-BFGS (Limited-memory BFGS) solves this by:

- Never storing the full Hessian.
- Maintaining a rolling history of gradient and parameter changes to approximate $H_t^{-1}$ efficiently.

#### The Update Rule (Conceptual Form)

Given gradients $g_t$ and parameter steps $s_t = \theta_{t+1} - \theta_t$:

1. Compute gradient change:
   $$
   y_t = g_{t+1} - g_t
   $$

2. Update Hessian inverse estimate $H_{t+1}$ using rank-2 correction (BFGS rule):

   $$
   H_{t+1} = (I - \rho_t s_t y_t^\top) H_t (I - \rho_t y_t s_t^\top) + \rho_t s_t s_t^\top
   $$

   where $\rho_t = \frac{1}{y_t^\top s_t}$.

3. Use $H_{t+1}$ to compute the next update direction:
   $$
   p_t = -H_{t+1} g_{t+1}
   $$

L-BFGS keeps only a few $(s_t, y_t)$ pairs (e.g. last 10) to reduce memory from $O(d^2)$ to $O(md)$.

#### How Does It Work (Plain Language)?

Think of L-BFGS as a smart gradient descent:

- It "remembers" how the gradient changes as you move.
- From those changes, it infers the curvature (how steep or flat), like building a local map.
- It then adjusts step sizes per direction for faster progress.

You don't need the full Hessian, just the history of steps.

#### Step-by-Step Summary

| Step | Action                     | Description                                                       |
| ---- | -------------------------- | ----------------------------------------------------------------- |
| 1    | Compute gradient $g_t$     | Evaluate at current $\theta_t$                                    |
| 2    | Determine search direction | $p_t = -H_t g_t$                                                  |
| 3    | Line search                | Find step size $\alpha_t$ minimizing $L(\theta_t + \alpha_t p_t)$ |
| 4    | Update parameters          | $\theta_{t+1} = \theta_t + \alpha_t p_t$                          |
| 5    | Store history              | $(s_t, y_t)$ pairs for next iteration                             |
| 6    | Update inverse Hessian     | Using limited-memory formula                                      |

#### Example

Suppose you minimize $L(\theta) = \theta_1^2 + 10 \theta_2^2$ (an elongated bowl).
Vanilla gradient descent zigzags along the steep axis, converging slowly.
L-BFGS estimates curvature and preconditions the gradient, taking direct diagonal paths toward the minimum.

#### Tiny Code (Easy Versions)

Python (using SciPy)

```python
import numpy as np
from scipy.optimize import fmin_l_bfgs_b

def loss(theta):
    return np.sum(theta  2), 2 * theta  # return (loss, gradient)

theta0 = np.array([5.0, 1.0])
theta_opt, f_min, info = fmin_l_bfgs_b(loss, theta0)
print("Optimal θ:", theta_opt)
```

C (Outline)

```c
// Pseudocode only: implementing L-BFGS manually requires line search and history buffers
// 1. Initialize theta, grad
// 2. Maintain arrays for s[i] = delta_theta, y[i] = delta_grad (limited m history)
// 3. Two-loop recursion to compute search direction
// 4. Perform line search to find optimal step size
// 5. Update theta, store new s, y
```

#### Why It Matters

- Faster convergence than first-order methods.
- No Hessian needed, uses gradient differences.
- Ideal for convex, smooth functions.
- Common in logistic regression, SVMs, and classical ML (before deep nets).

In deep learning, L-BFGS is used for fine-tuning or small networks (where full batches are feasible).

#### A Gentle Proof (Why It Works)

BFGS ensures positive definiteness of $H_t$ if $y_t^\top s_t > 0$, guaranteeing descent direction:

$$
g_t^\top p_t = -g_t^\top H_t g_t < 0
$$

Thus, each step moves downhill.

L-BFGS approximates the Newton step:

$$
\Delta \theta = -H_t^{-1} g_t
$$

using a compact two-loop recursion that reconstructs $H_t g_t$ from stored $(s, y)$ pairs.

#### Try It Yourself

1. Compare L-BFGS vs SGD on quadratic $L(\theta) = \theta_1^2 + 100\theta_2^2$.
2. Visualize trajectories, L-BFGS converges in fewer steps.
3. Vary history size $m = 3, 5, 10$.
4. Try with and without line search.
5. Use `fmin_l_bfgs_b` in SciPy for logistic regression.

#### Test Cases

| Function        | Optimizer | Behavior                            |
| --------------- | --------- | ----------------------------------- |
| Quadratic bowl  | L-BFGS    | Fast convergence                    |
| Ill-conditioned | L-BFGS    | Adjusts curvature                   |
| Non-convex      | L-BFGS    | May get stuck in local minima       |
| Deep net        | L-BFGS    | Works only with full-batch training |

#### Complexity

- Per iteration: $O(md)$ (with memory $m$)
- Memory: $O(md)$
- Convergence: superlinear for convex smooth functions

L-BFGS is a "memory-efficient Newton's method", harnessing curvature from history, not Hessians, to speed up convergence on smooth terrains.

# Section 94. Deep Learning 

### 931. Backpropagation

Backpropagation is the cornerstone algorithm for training neural networks. It efficiently computes gradients of the loss with respect to every weight by applying the chain rule of calculus through the network, allowing gradient-based optimizers like SGD or Adam to update parameters.

#### What Problem Are We Solving?

In neural networks, we want to minimize a loss function
$$
L(\theta)
$$
over parameters $\theta$ (weights and biases).
Directly computing $\frac{\partial L}{\partial \theta}$ for every parameter by hand is infeasible, too many dependencies, too many paths.

Backpropagation provides a systematic, efficient way to compute all gradients in one backward pass using the chain rule.

#### The Core Idea

Every layer in the network transforms an input:
$$
a^{(l)} = f^{(l)}(W^{(l)} a^{(l-1)} + b^{(l)})
$$

Loss $L$ depends on the final output $a^{(L)}$.
By applying the chain rule, we propagate derivatives backward from the output to each layer.

For each layer $l$, we compute:

1. Output error:
   $$
   \delta^{(L)} = \nabla_{a^{(L)}} L \odot f'^{(L)}(z^{(L)})
   $$

2. Backward recursion:
   $$
   \delta^{(l)} = (W^{(l+1)})^\top \delta^{(l+1)} \odot f'^{(l)}(z^{(l)})
   $$

3. Gradients:
   $$
   \frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} (a^{(l-1)})^\top, \quad
   \frac{\partial L}{\partial b^{(l)}} = \delta^{(l)}
   $$

Then update parameters with any gradient-based optimizer.

#### How Does It Work (Plain Language)?

Think of backpropagation like blaming errors:

- The output layer compares its prediction to the target and computes an error.
- Each hidden layer asks, "How much did I contribute to that error?"
  It passes credit (or blame) backward through connections.

By repeating this, each neuron learns how to adjust its weights to reduce future mistakes.

#### Step-by-Step Summary

| Step | Action                      | Description                           |
| ---- | --------------------------- | ------------------------------------- |
| 1    | Forward Pass            | Compute activations layer by layer    |
| 2    | Compute Loss            | Compare prediction to target          |
| 3    | Initialize Output Error | Derivative of loss w.r.t. final layer |
| 4    | Backward Pass           | Apply chain rule layer by layer       |
| 5    | Compute Gradients       | For all weights and biases            |
| 6    | Update Parameters       | Using optimizer (e.g. SGD, Adam)      |

#### Example

For a 2-layer network:
$$
a^{(1)} = \sigma(W^{(1)} x + b^{(1)})
$$
$$
a^{(2)} = \text{softmax}(W^{(2)} a^{(1)} + b^{(2)})
$$
$$
L = \text{cross\_entropy}(a^{(2)}, y)
$$

Then:

- Output layer error:
  $$
  \delta^{(2)} = a^{(2)} - y
  $$
- Hidden layer error:
  $$
  \delta^{(1)} = (W^{(2)})^\top \delta^{(2)} \odot \sigma'(z^{(1)})
  $$

#### Tiny Code (Easy Versions)

Python

```python
import numpy as np

# Simple 1-hidden-layer NN
def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_deriv(x): return x * (1 - x)

# Forward
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
W1 = np.random.randn(2, 2)
b1 = np.zeros((1, 2))
W2 = np.random.randn(2, 1)
b2 = np.zeros((1, 1))
lr = 0.1

for epoch in range(1000):
    # Forward
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)
    loss = np.mean((y - a2)  2)
    
    # Backward
    dL_da2 = -(y - a2)
    da2_dz2 = sigmoid_deriv(a2)
    dz2_dW2 = a1
    delta2 = dL_da2 * da2_dz2
    
    dW2 = dz2_dW2.T @ delta2
    db2 = np.sum(delta2, axis=0, keepdims=True)
    
    delta1 = (delta2 @ W2.T) * sigmoid_deriv(a1)
    dW1 = X.T @ delta1
    db1 = np.sum(delta1, axis=0, keepdims=True)
    
    # Update
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2
```

C (Outline)

```c
// 1. Forward pass: compute activations
// 2. Compute loss and output error
// 3. Backward pass: propagate errors using chain rule
// 4. Compute weight gradients
// 5. Update weights: W -= lr * grad
```

#### Why It Matters

- Efficient: computes all gradients in $O(\text{\#parameters})$.
- Scalable: works for any differentiable network.
- Foundation of deep learning: every modern network, CNNs, RNNs, Transformers, uses it.
- Enables automatic differentiation (autograd).

Backpropagation turned neural networks from theory to practice, enabling the modern AI revolution.

#### A Gentle Proof (Why It Works)

For composed functions
$$
f(x) = f_3(f_2(f_1(x)))
$$
By the chain rule:
$$
\frac{df}{dx} = f_3'(f_2(f_1(x))) \cdot f_2'(f_1(x)) \cdot f_1'(x)
$$

Backprop efficiently applies this recursively, caching intermediate results (activations) from the forward pass, avoiding recomputation.

#### Try It Yourself

1. Implement a 2-layer perceptron.
2. Compute gradients manually (small net) and compare with backprop.
3. Visualize $\delta$ values per layer.
4. Add ReLU activation and note simplicity of derivative.
5. Compare training speed with/without caching activations.

#### Test Cases

| Network   | Layers | Behavior                   |
| --------- | ------ | -------------------------- |
| Linear    | 1      | Gradients match analytical |
| MLP       | 2      | Smooth convergence         |
| Deep net  | 10     | Works with caching         |
| Nonlinear | ReLU   | Sparse gradients, stable   |

#### Complexity

- Forward pass: $O(P)$
- Backward pass: $O(P)$
- Memory: $O(P)$ for activations
  ($P$ = number of parameters)

Backpropagation is the heart of learning, applying calculus in reverse to teach machines how to adjust every connection toward better predictions.

### 932. Xavier / He Initialization

Neural networks learn by propagating signals forward and gradients backward. But if the weights are too large or too small, activations explode or vanish across layers.
Xavier and He initialization solve this by scaling the initial weights according to the number of input (and sometimes output) connections, keeping the signal's variance stable throughout the network.

#### What Problem Are We Solving?

When initializing neural network weights, two problems often occur:

1. Exploding activations:
   If weights are too large, activations grow exponentially with each layer.

2. Vanishing activations:
   If weights are too small, signals shrink to zero, gradients vanish.

We want the variance of activations to remain approximately constant across layers:
$$
\mathrm{Var}[a^{(l)}] \approx \mathrm{Var}[a^{(l-1)}]
$$

Xavier and He initializations use statistical reasoning to find the right scale.

#### The Core Idea

For a neuron:
$$
a^{(l)} = f(W^{(l)} a^{(l-1)} + b^{(l)})
$$

Let each input have variance $\mathrm{Var}[a^{(l-1)}] = v$,
and each weight have variance $\mathrm{Var}[W^{(l)}] = \sigma^2$.

If we want to preserve variance, we need:
$$
n_{\text{in}} \sigma^2 = 1
$$

So we set:
$$
\sigma^2 = \frac{1}{n_{\text{in}}}
$$

This gives Xavier (Glorot) Initialization.

#### Formulas

##### Xavier (Glorot) Initialization

For tanh or sigmoid activations (zero-centered):

- Normal distribution:
  $$
  W \sim \mathcal{N}\left(0, \frac{1}{n_{\text{in}}}\right)
  $$
- Uniform distribution:
  $$
  W \sim U\left[-\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}, \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}\right]
  $$

##### He Initialization

For ReLU activations (half the inputs zeroed):

- Normal:
  $$
  W \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}}}\right)
  $$
- Uniform:
  $$
  W \sim U\left[-\sqrt{\frac{6}{n_{\text{in}}}}, \sqrt{\frac{6}{n_{\text{in}}}}\right]
  $$

He initialization doubles the variance to account for ReLU's sparsity.

#### How Does It Work (Plain Language)?

Imagine passing water through a series of pipes (layers).
If some pipes widen and others shrink, the flow changes unpredictably.
Xavier and He make each layer's pipe diameter proportional to its number of inputs, keeping the "flow" (variance) consistent.

- Xavier: balances forward and backward flow for smooth gradient propagation.
- He: boosts variance for ReLU, since half the activations get zeroed.

#### Step-by-Step Summary

| Step | Action            | Description                             |
| ---- | ----------------- | --------------------------------------- |
| 1    | Choose activation | `tanh` → Xavier, `ReLU` → He            |
| 2    | Count fan-in      | $n_{\text{in}} = \text{\# inputs}$       |
| 3    | Sample weights    | From scaled normal/uniform distribution |
| 4    | Initialize biases | Typically zeros                         |
| 5    | Start training    | Activations maintain healthy variance   |

#### Example

Suppose a layer has 256 inputs and 128 outputs.

- Xavier (tanh):
  $$
  \text{std} = \sqrt{\frac{2}{256 + 128}} = 0.05
  $$
  $$
  W \sim \mathcal{N}(0, 0.05^2)
  $$

- He (ReLU):
  $$
  \text{std} = \sqrt{\frac{2}{256}} = 0.088
  $$
  $$
  W \sim \mathcal{N}(0, 0.088^2)
  $$

#### Tiny Code (Easy Versions)

Python (NumPy)

```python
import numpy as np

def xavier_init(n_in, n_out):
    limit = np.sqrt(6 / (n_in + n_out))
    return np.random.uniform(-limit, limit, (n_in, n_out))

def he_init(n_in, n_out):
    std = np.sqrt(2 / n_in)
    return np.random.randn(n_in, n_out) * std

# Example
W1 = xavier_init(256, 128)
W2 = he_init(256, 128)
```

C (Outline)

```c
#include <math.h>
#include <stdlib.h>

double rand_uniform(double a, double b) {
    return a + (b - a) * ((double) rand() / RAND_MAX);
}

void xavier_init(double* W, int n_in, int n_out) {
    double limit = sqrt(6.0 / (n_in + n_out));
    for (int i = 0; i < n_in * n_out; i++)
        W[i] = rand_uniform(-limit, limit);
}
```

#### Why It Matters

- Prevents vanishing/exploding activations.
- Keeps gradient variance stable across layers.
- Essential for deep networks (10+ layers).
- Simple change → massive improvement in training stability.

Without proper initialization, even the best optimizers fail to converge.

#### A Gentle Proof (Why It Works)

Variance of a neuron's output:
$$
\mathrm{Var}[a^{(l)}] = n_{\text{in}} \mathrm{Var}[W^{(l)}] \mathrm{Var}[a^{(l-1)}]
$$

To maintain $\mathrm{Var}[a^{(l)}] = \mathrm{Var}[a^{(l-1)}]$, set:
$$
\mathrm{Var}[W^{(l)}] = \frac{1}{n_{\text{in}}}
$$

ReLU halves the effective variance, so multiply by 2 → He initialization.

#### Try It Yourself

1. Train a deep MLP with random small weights → observe vanishing gradients.
2. Repeat with Xavier → stable learning.
3. Try ReLU with He → faster convergence.
4. Compare variance of activations layer-by-layer.
5. Plot histograms of activations during forward pass.

#### Test Cases

| Activation | Method       | Result                   |
| ---------- | ------------ | ------------------------ |
| tanh       | Xavier       | Balanced activations     |
| sigmoid    | Xavier       | Stable early layers      |
| ReLU       | He           | Non-zero stable variance |
| LeakyReLU  | He           | Works well               |
| None       | Random small | Vanishing gradients      |

#### Complexity

- Time: $O(P)$ (one-time setup)
- Space: $O(P)$ (weights)
- Benefit: stabilizes signal propagation from the start

Proper initialization is step 0 of learning, Xavier and He set the stage so your optimizer doesn't have to fight bad scaling before it can learn.

### 933. Dropout

Dropout is a regularization technique for neural networks that prevents overfitting by randomly turning off neurons during training.
Each training pass samples a smaller subnetwork, forcing the model to rely on distributed representations rather than memorizing patterns.

#### What Problem Are We Solving?

Deep networks with millions of parameters can easily overfit, they memorize training data instead of learning general patterns.
Even with early stopping or weight decay, neurons can still co-adapt (depend too strongly on one another).

Dropout breaks these co-adaptations by making each neuron unreliable during training, like training an ensemble of smaller networks that must all perform well.

#### The Core Idea

During training, each neuron's output is randomly "dropped" with probability $p$.
Formally, for activation $a_i^{(l)}$ in layer $l$:

$$
\tilde{a}_i^{(l)} = r_i^{(l)} \cdot a_i^{(l)}, \quad r_i^{(l)} \sim \text{Bernoulli}(1 - p)
$$

where:

- $p$ = dropout rate (e.g. 0.5)
- $r_i^{(l)}$ = random mask (0 or 1)

At inference time, no neurons are dropped, instead, activations are scaled by $(1 - p)$ to match expected output.

#### How Does It Work (Plain Language)?

Imagine a classroom where half the students are randomly asked to leave during each discussion.
The remaining students must still solve the problem, so everyone learns independently and can handle missing teammates later.

Dropout does the same for neurons:

- During training → random neurons go silent.
- During testing → everyone returns, but each neuron's output is scaled down.

This creates robustness and prevents reliance on any single feature.

#### Step-by-Step Summary

| Step | Action                                         | Description                                     |
| ---- | ---------------------------------------------- | ----------------------------------------------- |
| 1    | Choose dropout rate $p$                        | Common: 0.2–0.5                                 |
| 2    | Sample mask $r_i \sim \text{Bernoulli}(1 - p)$ | One per neuron                                  |
| 3    | Apply mask                                     | $\tilde{a}_i = r_i \cdot a_i$                   |
| 4    | Forward pass                                   | Use masked activations                          |
| 5    | Backprop                                       | Gradients flow only through active neurons      |
| 6    | Inference                                      | Use all neurons, scale activations by $(1 - p)$ |

#### Example

For a hidden layer:
$$
a^{(l)} = f(W^{(l)} a^{(l-1)} + b^{(l)})
$$
Apply dropout:
$$
r^{(l)} \sim \text{Bernoulli}(1 - p)
$$
$$
\tilde{a}^{(l)} = r^{(l)} \odot a^{(l)}
$$

At inference:
$$
a^{(l)}*{\text{test}} = (1 - p) a^{(l)}*{\text{train}}
$$

#### Tiny Code (Easy Versions)

Python

```python
import numpy as np

def dropout_layer(a, p=0.5, train=True):
    if train:
        mask = (np.random.rand(*a.shape) > p).astype(float)
        return a * mask / (1 - p)
    else:
        return a  # scaled during training
```

C (Outline)

```c
#include <stdlib.h>

void dropout(double* a, int n, double p) {
    for (int i = 0; i < n; i++) {
        double r = (double) rand() / RAND_MAX;
        a[i] = (r > p) ? a[i] / (1.0 - p) : 0.0;
    }
}
```

#### Why It Matters

- Reduces overfitting: prevents reliance on single neurons.
- Acts as ensemble averaging: each training iteration samples a subnetwork.
- Improves generalization: especially for fully connected layers.
- Simple and effective: no extra parameters or complexity.

Dropout was one of the major breakthroughs enabling deep neural networks to generalize well.

#### A Gentle Proof (Why It Works)

The expected activation stays constant:

$$
\mathbb{E}[\tilde{a}_i] = (1 - p)a_i
$$

To preserve magnitude, we scale during training by $\frac{1}{1 - p}$:
$$
\tilde{a}_i = \frac{r_i}{1 - p} a_i
$$

Thus:
$$
\mathbb{E}[\tilde{a}_i] = a_i
$$

This keeps mean activations unchanged across training and inference.

#### Try It Yourself

1. Train a small MLP on MNIST with and without dropout.
2. Compare train and test accuracy.
3. Experiment with $p = 0.2, 0.5, 0.8$.
4. Apply dropout only on hidden layers.
5. Observe slower training but better generalization.

#### Test Cases

| Network  | Dropout Rate | Effect              |
| -------- | ------------ | ------------------- |
| 0 (None) | 0.0          | Overfitting         |
| Moderate | 0.5          | Best generalization |
| High     | 0.8          | Underfitting        |
| ConvNet  | 0.3          | Stable results      |

#### Complexity

- Time: $O(N)$ per layer (mask sampling)
- Memory: $O(N)$ (mask storage)
- Inference cost: none

Dropout makes your network forget, just enough to remember the right things.

### 934. Batch Normalization

Batch Normalization (BatchNorm) stabilizes and accelerates training by normalizing activations across each mini-batch.
It keeps layer outputs well-behaved, neither exploding nor vanishing, so networks can train deeper, faster, and with higher learning rates.

#### What Problem Are We Solving?

During training, as earlier layers change, the input distribution to later layers also shifts, a problem called internal covariate shift.
This makes optimization harder because each layer must constantly adapt to the changing distribution of its inputs.

BatchNorm reduces this drift by normalizing layer activations to have zero mean and unit variance, then allowing the network to re-learn an appropriate scale and offset.

#### The Core Idea

For each mini-batch and feature channel:

$$
\mu_B = \frac{1}{m} \sum_{i=1}^m x_i
$$
$$
\sigma_B^2 = \frac{1}{m} \sum_{i=1}^m (x_i - \mu_B)^2
$$

Normalize and re-scale:

$$
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
$$
$$
y_i = \gamma \hat{x}_i + \beta
$$

where:

- $\gamma$ and $\beta$ are learned parameters for scale and shift.
- $\epsilon$ prevents division by zero.

At inference, moving averages of $\mu_B$ and $\sigma_B^2$ are used instead of batch statistics.

#### How Does It Work (Plain Language)?

Imagine each neuron's output as students' test scores, some high, some low, with wildly different averages in every class.
BatchNorm centers (subtract mean) and scales (divide by std) those scores, so the next layer always receives values in a similar range.
Then it lets the model reintroduce personality with learned $\gamma$ and $\beta$.

This makes the optimization landscape smoother and easier to traverse.

#### Step-by-Step Summary

| Step | Action                    | Description                               |
| ---- | ------------------------- | ----------------------------------------- |
| 1    | Compute mean and variance | From mini-batch                           |
| 2    | Normalize                 | Subtract mean, divide by std              |
| 3    | Scale and shift           | Apply learned $\gamma$, $\beta$           |
| 4    | Forward pass              | Use normalized activations                |
| 5    | Backpropagate             | Compute gradients for $\gamma$, $\beta$   |
| 6    | Inference                 | Use running averages of $\mu$, $\sigma^2$ |

#### Example

For a layer with input $x = [x_1, x_2, \dots, x_m]$:

$$
\mu_B = \frac{1}{m} \sum_i x_i, \quad
\sigma_B^2 = \frac{1}{m} \sum_i (x_i - \mu_B)^2
$$

Then:

$$
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad
y_i = \gamma \hat{x}_i + \beta
$$

At test time:
$$
y_i = \frac{x_i - \mu_{\text{running}}}{\sqrt{\sigma_{\text{running}}^2 + \epsilon}} \gamma + \beta
$$

#### Tiny Code (Easy Versions)

Python (NumPy)

```python
import numpy as np

def batchnorm_forward(x, gamma, beta, eps=1e-5):
    mu = np.mean(x, axis=0)
    var = np.var(x, axis=0)
    x_hat = (x - mu) / np.sqrt(var + eps)
    y = gamma * x_hat + beta
    return y
```

C (Outline)

```c
#include <math.h>

void batchnorm(double* x, double* y, int n, double gamma, double beta, double eps) {
    double mean = 0.0, var = 0.0;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;
    for (int i = 0; i < n; i++) var += (x[i] - mean) * (x[i] - mean);
    var /= n;
    for (int i = 0; i < n; i++)
        y[i] = gamma * ((x[i] - mean) / sqrt(var + eps)) + beta;
}
```

#### Why It Matters

- Faster convergence: allows higher learning rates.
- Improved stability: keeps gradients in a healthy range.
- Regularization effect: slight noise from batch stats reduces overfitting.
- Easier initialization: less sensitive to starting weights.

BatchNorm became the default normalization in CNNs, RNNs, and Transformers before layer normalization took over for non-convolutional models.

#### A Gentle Proof (Why It Works)

Gradient explosion happens when activation variance grows with depth.
BatchNorm constrains variance to $\approx 1$, preventing blow-up.

For any neuron:
$$
\mathrm{Var}[\hat{x}] = \frac{\mathrm{Var}[x]}{\sigma_B^2 + \epsilon} \approx 1
$$

Thus, the gradient with respect to $x$ is also stabilized, leading to smoother optimization and more reliable updates.

#### Try It Yourself

1. Train a 5-layer MLP with and without BatchNorm.
2. Observe faster convergence with normalization.
3. Vary batch size, small batches make BatchNorm noisier.
4. Compare with LayerNorm (used in Transformers).
5. Inspect $\gamma$, $\beta$ after training, they adaptively re-scale outputs.

#### Test Cases

| Architecture | Normalization | Effect                     |
| ------------ | ------------- | -------------------------- |
| MLP          | None          | Slow convergence           |
| MLP          | BatchNorm     | Faster and stable          |
| CNN          | BatchNorm     | Smoother gradients         |
| Transformer  | LayerNorm     | Works better for sequences |

#### Complexity

- Time: $O(N)$ per batch
- Memory: $O(N)$ for storing mean, variance
- Inference: cached running averages

BatchNorm is one of the simplest yet most transformative tricks in deep learning, a normalization layer that turned unstable deep networks into trainable ones.

### 935. Layer Normalization

Layer Normalization (LayerNorm) normalizes activations within each sample, not across a batch.
It provides stable training for models where batch statistics are unreliable, especially RNNs, Transformers, and other sequence models.

#### What Problem Are We Solving?

Batch Normalization relies on mini-batch statistics.
But in tasks like sequence modeling, reinforcement learning, or variable-length inputs:

- Batch sizes can be very small.
- The same layer can process very different sequences.
- Batch statistics fluctuate too much.

LayerNorm solves this by normalizing activations across the features of each individual sample, making it independent of batch size.

#### The Core Idea

Given an input vector for one sample:
$$
x = [x_1, x_2, \dots, x_H]
$$
where $H$ is the number of hidden units (features).

Compute per-sample statistics:
$$
\mu = \frac{1}{H} \sum_{i=1}^H x_i
$$
$$
\sigma^2 = \frac{1}{H} \sum_{i=1}^H (x_i - \mu)^2
$$

Normalize and re-scale:
$$
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad
y_i = \gamma_i \hat{x}_i + \beta_i
$$

Here $\gamma_i$ and $\beta_i$ are learned per-feature scaling and shifting parameters.

#### How Does It Work (Plain Language)?

Think of each neuron's activations in a single input as a mini-ecosystem.
LayerNorm makes sure that, for every input, the activations have a consistent "climate" (mean 0, variance 1), regardless of the batch or sequence.

Instead of comparing across examples, it standardizes within each example.

This makes it robust for small batches, sequential data, and attention-based architectures.

#### Step-by-Step Summary

| Step | Action                                  | Description                         |
| ---- | --------------------------------------- | ----------------------------------- |
| 1    | Take one sample (vector of activations) | $x = [x_1, ..., x_H]$               |
| 2    | Compute mean and variance               | Over features, not batch            |
| 3    | Normalize                               | Subtract mean, divide by std        |
| 4    | Re-scale                                | Multiply by $\gamma$, add $\beta$   |
| 5    | Use output                              | Feed normalized activations forward |

#### Example

For a Transformer hidden state $x \in \mathbb{R}^{d_{\text{model}}}$:

$$
\mu = \frac{1}{d_{\text{model}}} \sum_i x_i, \quad
\sigma^2 = \frac{1}{d_{\text{model}}} \sum_i (x_i - \mu)^2
$$
$$
\text{LayerNorm}(x) = \gamma \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

#### Tiny Code (Easy Versions)

Python

```python
import numpy as np

def layernorm(x, gamma, beta, eps=1e-5):
    mu = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_hat = (x - mu) / np.sqrt(var + eps)
    return gamma * x_hat + beta
```

C (Outline)

```c
#include <math.h>

void layernorm(double* x, double* y, int H, double* gamma, double* beta, double eps) {
    double mean = 0.0, var = 0.0;
    for (int i = 0; i < H; i++) mean += x[i];
    mean /= H;
    for (int i = 0; i < H; i++) var += (x[i] - mean) * (x[i] - mean);
    var /= H;
    for (int i = 0; i < H; i++)
        y[i] = gamma[i] * ((x[i] - mean) / sqrt(var + eps)) + beta[i];
}
```

#### Why It Matters

- Works with any batch size, even 1.
- Stabilizes RNNs and Transformers where BatchNorm fails.
- Smooths optimization landscape and gradient flow.
- Adds minimal computational overhead.

LayerNorm is one of the key innovations that made Transformers (like GPT and BERT) train stably across long sequences and variable contexts.

#### A Gentle Proof (Why It Works)

For each sample:
$$
\mathbb{E}[\hat{x}] = 0, \quad \mathrm{Var}[\hat{x}] = 1
$$

Thus the normalized activations maintain consistent magnitude, ensuring gradients stay stable:
$$
\frac{\partial L}{\partial x_i} = \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}}
\left( \frac{\partial L}{\partial \hat{x}_i} -
\frac{1}{H} \sum_j \frac{\partial L}{\partial \hat{x}_j} -
\frac{\hat{x}_i}{H} \sum_j \frac{\partial L}{\partial \hat{x}_j} \hat{x}_j \right)
$$

This preserves zero-mean gradients, avoiding drift across layers.

#### Try It Yourself

1. Train a Transformer block with and without LayerNorm.
2. Compare loss stability, LayerNorm smooths training.
3. Try tiny batch sizes ($B=1$ or $B=2$).
4. Apply before or after residual connections.
5. Inspect learned $\gamma$, $\beta$, they adapt output range.

#### Test Cases

| Model       | Normalization | Result          |
| ----------- | ------------- | --------------- |
| MLP         | BatchNorm     | OK              |
| RNN         | BatchNorm     | Unstable        |
| RNN         | LayerNorm     | Stable          |
| Transformer | LayerNorm     | Standard choice |

#### Complexity

- Time: $O(H)$ per sample
- Memory: $O(H)$ for mean, variance
- Inference: identical cost

Layer Normalization is the steadying hand of modern deep networks, ensuring consistent activations, even when the batch is small or the sequence is long.

### 936. Gradient Clipping

Gradient Clipping is a stabilization technique that prevents gradients from becoming too large during backpropagation.
It's especially important in deep networks and RNNs, where exploding gradients can cause wild parameter updates and destroy training progress.

#### What Problem Are We Solving?

During training, gradients can sometimes explode, grow exponentially through many layers or time steps.
This happens when multiplying large weights repeatedly in the chain rule.

If gradients become huge:

- The optimizer takes giant steps, overshooting the minimum.
- The loss becomes NaN due to numerical overflow.
- Learning diverges completely.

Gradient clipping solves this by limiting the gradient's magnitude before updating weights.

#### The Core Idea

Let $g$ be the full gradient vector.
Compute its L2 norm:
$$
|g|_2 = \sqrt{\sum_i g_i^2}
$$

If $|g|_2$ exceeds a threshold $c$, scale it down:
$$
g' = g \cdot \frac{c}{|g|_2}
$$

Then use $g'$ for the update.

This ensures:
$$
|g'|_2 \le c
$$

#### How Does It Work (Plain Language)?

Imagine trying to steer a car downhill, if you push the gas too hard, you'll spin out of control.
Gradient clipping is like setting a speed limiter: it allows motion, but caps it at a safe velocity.

If gradients are small → do nothing.
If they explode → scale them back smoothly to the target range.

#### Step-by-Step Summary

| Step | Action                     | Description                       |
| ---- | -------------------------- | --------------------------------- |
| 1    | Compute gradients          | $g = \nabla_\theta L$             |
| 2    | Compute norm               | $|g|_2 = \sqrt{\sum_i g_i^2}$     |
| 3    | Compare with threshold $c$ | Typical $c = 1.0$ or $5.0$        |
| 4    | If too large, rescale      | $g \gets g \cdot \frac{c}{|g|_2}$ |
| 5    | Update parameters          | $\theta \gets \theta - \eta g$    |

#### Example

Suppose:
$$
g = [3, 4], \quad |g|_2 = 5
$$
and threshold $c = 2$.

Then:
$$
g' = g \cdot \frac{2}{5} = [1.2, 1.6]
$$
Now $|g'|_2 = 2$, safely within the limit.

#### Variants

1. Global norm clipping
   Applies the same scaling factor to all parameters.

   $$
   g' = g \cdot \frac{c}{\max(|g|_2, c)}
   $$

2. Per-layer or per-parameter clipping
   Each tensor's gradient is clipped individually.

3. Value clipping
   Clamp each gradient component:
   $$
   g_i' = \text{clip}(g_i, -c, c)
   $$

#### Tiny Code (Easy Versions)

Python

```python
import numpy as np

def clip_gradients(grads, clip_value=1.0):
    norm = np.sqrt(sum(np.sum(g  2) for g in grads))
    if norm > clip_value:
        scale = clip_value / norm
        grads = [g * scale for g in grads]
    return grads
```

C (Outline)

```c
#include <math.h>

void clip_gradient(double* g, int n, double c) {
    double norm = 0.0;
    for (int i = 0; i < n; i++) norm += g[i] * g[i];
    norm = sqrt(norm);
    if (norm > c) {
        double scale = c / norm;
        for (int i = 0; i < n; i++) g[i] *= scale;
    }
}
```

#### Why It Matters

- Prevents exploding gradients, especially in RNNs.
- Makes training stable for deep networks.
- Allows larger learning rates safely.
- Used with optimizers like SGD, Adam, and RMSProp.

Without clipping, long-sequence models like LSTMs and Transformers would be nearly impossible to train reliably.

#### A Gentle Proof (Why It Works)

Clipping constrains the update step size:

$$
|\Delta \theta|_2 = \eta |g'|_2 \le \eta c
$$

Thus, even if raw gradients explode, the parameter change per step remains bounded, preventing numerical instability.

It doesn't bias directions significantly (since scaling is uniform), only magnitudes.

#### Try It Yourself

1. Train an RNN on long sequences with and without clipping.
2. Observe loss divergence when not clipped.
3. Try $c = 0.1, 1.0, 5.0$.
4. Plot $|g|_2$ per step, clipping caps spikes.
5. Combine with Adam, clipping still effective.

#### Test Cases

| Model       | Threshold $c$ | Result              |
| ----------- | ------------- | ------------------- |
| RNN         | None          | Exploding gradients |
| RNN         | 1.0           | Stable              |
| LSTM        | 5.0           | Smooth training     |
| Transformer | 1.0           | Standard setup      |

#### Complexity

- Time: $O(P)$ (sum and rescale)
- Memory: negligible
- Effect: stabilizes updates without harming convergence

Gradient clipping is a small, almost invisible trick, but it's the difference between chaos and control in deep learning training.

### 937. Early Stopping

Early Stopping is a simple yet powerful regularization technique that halts training when performance on a validation set stops improving.
It prevents overfitting by saving the model at its best generalization point, before it starts memorizing the training data.

#### What Problem Are We Solving?

When training deep networks, loss on the training set usually keeps decreasing,
but loss on the validation set may start to rise after a certain number of epochs.
This is the point where the model begins to overfit, learning noise instead of structure.

Instead of guessing the right number of epochs, Early Stopping automatically detects this turning point and freezes the model there.

#### The Core Idea

Monitor validation loss $L_{\text{val}}$ over epochs:

1. Keep track of the best validation loss so far:
   $$
   L^* = \min_t L_{\text{val}}(t)
   $$

2. If loss hasn't improved for $p$ epochs (the patience parameter), stop training.

Formally:

$$
\text{if } L_{\text{val}}(t) > L^* - \delta \text{ for } p \text{ epochs, stop.}
$$

Here:

- $\delta$ = minimum improvement required
- $p$ = patience (tolerance before stopping)

#### How Does It Work (Plain Language)?

Think of training like baking bread.
If you bake too little, it's undercooked (underfitting).
If you bake too long, it burns (overfitting).
Early stopping watches the oven and stops at just the right moment, when validation loss smells perfect.

It's a form of implicit regularization, no penalty term, just an intelligent pause.

#### Step-by-Step Summary

| Step | Action                 | Description                                    |
| ---- | ---------------------- | ---------------------------------------------- |
| 1    | Split data             | Training + validation sets                     |
| 2    | Train model            | Update weights per epoch                       |
| 3    | Evaluate on validation | Compute $L_{\text{val}}$                       |
| 4    | Track best score       | Save model when validation improves            |
| 5    | Apply patience         | Wait a few epochs before stopping              |
| 6    | Restore best weights   | Revert to snapshot with lowest validation loss |

#### Example

Suppose we train for 100 epochs:

| Epoch | Train Loss | Val Loss | Action    |
| ----- | ---------- | -------- | --------- |
| 1–10  | ↓          | ↓        | Improving |
| 11–30 | ↓          | Plateau  | Watch     |
| 31–40 | ↓          | ↑        | Stop!     |

If patience = 10, we stop after epoch 40 and restore weights from epoch 30, the best generalization point.

#### Tiny Code (Easy Versions)

Python

```python
best_val = float('inf')
patience, wait = 10, 0
best_weights = None

for epoch in range(100):
    train_one_epoch(model)
    val_loss = evaluate(model)
    if val_loss < best_val - 1e-4:
        best_val = val_loss
        best_weights = model.copy_weights()
        wait = 0
    else:
        wait += 1
    if wait >= patience:
        print("Early stopping at epoch", epoch)
        model.load_weights(best_weights)
        break
```

C (Outline)

```c
double best_val = 1e9;
int patience = 10, wait = 0;

for (int epoch = 0; epoch < 100; epoch++) {
    double val_loss = evaluate_model();
    if (val_loss < best_val - 1e-4) {
        best_val = val_loss;
        save_model();
        wait = 0;
    } else {
        wait++;
    }
    if (wait >= patience) {
        printf("Early stopping at epoch %d\n", epoch);
        load_best_model();
        break;
    }
}
```

#### Why It Matters

- Prevents overfitting without tuning extra hyperparameters.
- Automatically picks the best epoch for generalization.
- Reduces training time, stops before diminishing returns.
- Common in deep learning frameworks (e.g. `Keras.callbacks.EarlyStopping`).

#### A Gentle Proof (Why It Works)

Generalization error $E_g$ typically follows a U-shaped curve:

$$
E_g(t) = E_{\text{train}}(t) + \text{overfit}(t)
$$

Early stopping halts before the overfit term dominates, capturing the minimum of $E_g(t)$:
$$
t^* = \arg\min_t E_{\text{val}}(t)
$$

This acts like L2 regularization in effect, it limits the total weight growth by limiting optimization time.

#### Try It Yourself

1. Train a small MLP on MNIST with patience = 5, 10, 20.
2. Plot training and validation loss.
3. Watch how longer patience slightly increases overfitting.
4. Combine with dropout for extra regularization.
5. Compare with L2 weight decay, similar effects, different mechanisms.

#### Test Cases

| Model       | Early Stopping | Result                      |
| ----------- | -------------- | --------------------------- |
| MLP         | None           | Overfits after ~40 epochs   |
| MLP         | Patience=10    | Best validation at epoch 30 |
| CNN         | Patience=5     | Stops early, best accuracy  |
| Transformer | Enabled        | Stable convergence          |

#### Complexity

- Time: saves time by halting early
- Memory: $O(P)$ (snapshot of best weights)
- Hyperparameters: patience, min delta

Early Stopping is the simplest kind of wisdom in machine learning, it doesn't fight overfitting, it just knows when to walk away.

### 938. Weight Decay

Weight Decay (also known as L2 Regularization) penalizes large parameter values by adding a small term to the loss function that discourages overly complex models.
It gently "shrinks" weights toward zero, preventing overfitting and improving generalization.

#### What Problem Are We Solving?

In neural networks, large weights can cause the model to memorize training data or become unstable during optimization.
We want the model to remain smooth, where small changes in input don't cause large swings in output.

Weight decay adds a penalty proportional to the squared magnitude of the weights, steering optimization toward simpler solutions.

#### The Core Idea

We modify the objective function from just the data loss $L(\theta)$ to include a regularization term:

$$
L_{\text{total}}(\theta) = L(\theta) + \frac{\lambda}{2} |\theta|_2^2
$$

where:

- $\theta$ = vector of all model parameters
- $\lambda$ = weight decay coefficient (regularization strength)

The gradient update becomes:

$$
\theta \gets \theta - \eta (\nabla_\theta L(\theta) + \lambda \theta)
$$

The extra term $\lambda \theta$ acts like a spring, pulling weights gently toward zero at each step.

#### How Does It Work (Plain Language)?

Imagine your weights as rubber bands stretched in different directions by training.
Weight decay constantly tugs them back toward the origin, not hard enough to stop learning, just enough to prevent them from stretching too far.

The result:

- Simpler functions
- Smoother decision boundaries
- Better generalization

#### Step-by-Step Summary

| Step | Action           | Description                                                         |
| ---- | ---------------- | ------------------------------------------------------------------- |
| 1    | Define loss      | $L_{\text{total}} = L + \frac{\lambda}{2}|\theta|^2$                |
| 2    | Compute gradient | $\nabla_\theta L_{\text{total}} = \nabla_\theta L + \lambda \theta$ |
| 3    | Update weights   | $\theta \gets \theta - \eta \nabla_\theta L_{\text{total}}$         |
| 4    | Repeat           | Continue training with regularized updates                          |

#### Example

Suppose:
$$
L(\theta) = (y - \theta x)^2, \quad \lambda = 0.1
$$

Then the total loss is:
$$
L_{\text{total}}(\theta) = (y - \theta x)^2 + 0.05 \theta^2
$$

Gradient:
$$
\frac{dL_{\text{total}}}{d\theta} = -2x(y - \theta x) + 0.1 \theta
$$

So even if the data term is small, $\theta$ will still slowly shrink toward zero.

#### Tiny Code (Easy Versions)

Python

```python
import numpy as np

def weight_decay_update(theta, grad, lr=0.01, wd=0.1):
    return theta - lr * (grad + wd * theta)

# Example
theta = np.array([1.0, -2.0])
grad = np.array([0.5, -0.5])
new_theta = weight_decay_update(theta, grad)
print(new_theta)
```

C (Outline)

```c
void weight_decay_update(double* theta, double* grad, int n, double lr, double wd) {
    for (int i = 0; i < n; i++) {
        theta[i] -= lr * (grad[i] + wd * theta[i]);
    }
}
```

#### Why It Matters

- Prevents overfitting by penalizing large weights.
- Improves generalization and smooths decision boundaries.
- Works well with SGD and most optimizers.
- Helps optimization by avoiding sharp minima.

Note: In AdamW, weight decay is applied *separately* (decoupled) from the gradient update for better consistency.

#### A Gentle Proof (Why It Works)

Regularization modifies the optimization trajectory.

For quadratic loss:
$$
\min_\theta |X\theta - y|^2 + \lambda |\theta|^2
$$
has the closed-form solution:
$$
\theta^* = (X^\top X + \lambda I)^{-1} X^\top y
$$

Here, $\lambda I$ dampens directions with small eigenvalues, reducing sensitivity to noise, equivalent to ridge regression.

#### Try It Yourself

1. Train a linear regression with and without weight decay.
2. Plot weights, with decay, they remain smaller and smoother.
3. Tune $\lambda$:

   * Too small → overfit
   * Too large → underfit
4. Combine with dropout or early stopping.
5. Observe training vs validation accuracy difference.

#### Test Cases

| Model             | Regularization | Effect                |
| ----------------- | -------------- | --------------------- |
| Linear Regression | None           | Overfits noise        |
| Linear Regression | L2 (λ=0.1)     | Smooths weights       |
| CNN               | λ=1e-4         | Better generalization |
| Transformer       | λ=0.01         | Common default        |

#### Complexity

- Time: negligible ($O(P)$)
- Memory: negligible (same parameter size)
- Effect: gradual shrinkage toward smoother models

Weight decay is the quiet discipline of neural networks, a small tug that keeps learning balanced between flexibility and restraint.

### 939. Learning Rate Scheduling

Learning Rate Scheduling dynamically adjusts the optimizer's learning rate during training to balance exploration (large steps) and convergence (small steps).
It is a crucial technique for achieving faster convergence, smoother optimization, and better generalization.

#### What Problem Are We Solving?

A fixed learning rate $\eta$ can cause problems:

- If $\eta$ is too high, training oscillates or diverges.
- If $\eta$ is too low, convergence is painfully slow or stuck in local minima.

The solution: start with a large learning rate for exploration and gradually decrease it as training stabilizes.
This allows the optimizer to take big steps early, then refine carefully near minima.

#### The Core Idea

We modify the learning rate $\eta_t$ as a function of time (epoch or step):

$$
\theta_{t+1} = \theta_t - \eta_t \nabla_\theta L(\theta_t)
$$

where $\eta_t$ is updated by a schedule function.

Common schedules:

| Schedule              | Formula                                                                         | Behavior                    |
| --------------------- | ------------------------------------------------------------------------------- | --------------------------- |
| Step Decay        | $\eta_t = \eta_0 \cdot \gamma^{\lfloor t/s \rfloor}$                            | Drop every $s$ epochs       |
| Exponential Decay | $\eta_t = \eta_0 e^{-kt}$                                                       | Smooth exponential decrease |
| Cosine Annealing  | $\eta_t = \eta_{\min} + \frac{1}{2}(\eta_0 - \eta_{\min})(1 + \cos(\pi t / T))$ | Gradual warm-to-cool        |
| Linear Decay      | $\eta_t = \eta_0 (1 - t/T)$                                                     | Simple linear decline       |
| Cyclical (CLR)    | Oscillates between $\eta_{\min}$ and $\eta_{\max}$                              | Periodic exploration        |
| Warmup            | Gradually increase $\eta_t$ in early epochs                                     | Prevents instability        |

#### How Does It Work (Plain Language)?

Think of training like hiking down a mountain.
At first, you take big confident steps to move quickly.
As you approach the valley, you slow down, adjusting carefully to avoid overshooting the minimum.

Learning rate scheduling does the same, it controls how fast you "walk" through the loss landscape.

#### Step-by-Step Summary

| Step | Action                             | Description                     |
| ---- | ---------------------------------- | ------------------------------- |
| 1    | Choose base learning rate $\eta_0$ | Often 0.1 or 0.001              |
| 2    | Select schedule type               | Step, exponential, cosine, etc. |
| 3    | Update $\eta_t$ at each step       | According to chosen rule        |
| 4    | Pass $\eta_t$ to optimizer         | Adjust step size dynamically    |
| 5    | Monitor training                   | Verify stable convergence       |

#### Example: Step Decay

$$
\eta_t = \eta_0 \cdot \gamma^{\lfloor t/s \rfloor}
$$

If $\eta_0 = 0.1$, $\gamma = 0.5$, $s = 10$:

| Epoch | Learning Rate |
| ----- | ------------- |
| 0–9   | 0.1           |
| 10–19 | 0.05          |
| 20–29 | 0.025         |

#### Example: Cosine Annealing

Smoothly decreases and restarts periodically:

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_0 - \eta_{\min})(1 + \cos(\pi t / T))
$$

Gives a gentle oscillation, large early steps, small near minima, then restarts for exploration.

#### Tiny Code (Easy Versions)

Python

```python
import numpy as np

def cosine_annealing_lr(t, T, eta_min=1e-5, eta_max=1e-2):
    return eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * t / T))

# Example
for epoch in range(0, 100, 10):
    print(epoch, cosine_annealing_lr(epoch, 100))
```

C (Outline)

```c
#include <math.h>

double cosine_annealing_lr(int t, int T, double eta_min, double eta_max) {
    return eta_min + 0.5 * (eta_max - eta_min) * (1 + cos(M_PI * t / T));
}
```

#### Why It Matters

- Improves convergence, faster and smoother than constant $\eta$.
- Reduces overfitting, smaller steps near minima generalize better.
- Enables large initial learning rates safely.
- Used in most modern training pipelines (e.g. ResNet, Transformers).

Schedulers are not just about speed, they shape how the optimizer explores and settles in the loss surface.

#### A Gentle Proof (Why It Works)

In convex optimization, convergence speed depends on $\eta_t$:

$$
L(\theta_t) - L^* \le \frac{C}{\sum_{i=1}^t \eta_i}
$$

Thus, decreasing $\eta_t$ ensures the cumulative step sizes converge, stabilizing updates as the model nears the optimum.

For cosine schedules, periodic restarts avoid getting stuck in flat minima, providing implicit regularization.

#### Try It Yourself

1. Train a CNN with constant vs scheduled learning rates.
2. Compare convergence speed and validation accuracy.
3. Try:

   * Step decay ($\gamma=0.5$ every 10 epochs)
   * Cosine annealing
   * Cyclical (CLR)
4. Plot $\eta_t$ across epochs, visualize the schedule's shape.
5. Combine with warmup for Transformers (e.g. increase $\eta$ linearly for first 5% of steps).

#### Test Cases

| Schedule         | Model       | Behavior                    |
| ---------------- | ----------- | --------------------------- |
| Constant         | CNN         | Slow, plateau early         |
| Step Decay       | CNN         | Sharp convergence           |
| Cosine Annealing | ResNet      | Smooth decay, best accuracy |
| Cyclical         | Transformer | Recurrent exploration       |

#### Complexity

- Time: $O(1)$ per step (lightweight update)
- Memory: negligible
- Benefit: faster, more stable convergence

Learning rate scheduling is like giving your optimizer intuition, knowing when to rush, when to slow, and when to take a deep breath before climbing again.

### 940. Residual Connections

Residual Connections (also called skip connections) allow information to bypass one or more layers in a neural network.
They were introduced in ResNet (2015) to solve the problem of vanishing gradients and enable the successful training of very deep networks, sometimes hundreds or even thousands of layers deep.

#### What Problem Are We Solving?

As networks get deeper:

- Gradients can vanish or explode during backpropagation.
- Training error may increase even when the model has more capacity.
- Deeper networks may learn slower or get stuck in poor local minima.

Residual connections fix this by providing shortcut paths that let gradients flow more directly through the network.

#### The Core Idea

A standard layer learns a mapping:

$$
y = \mathcal{F}(x)
$$

A residual layer instead learns:

$$
y = \mathcal{F}(x) + x
$$

where:

- $\mathcal{F}(x)$ = residual mapping (e.g. convolution, batch norm, activation)
- $x$ = identity shortcut (input)

During backpropagation, gradients can flow both through $\mathcal{F}(x)$ and directly through $x$, avoiding vanishing.

#### How Does It Work (Plain Language)?

Imagine building a tower of blocks, each layer adds a modification to what came before.
Residual connections let some blocks pass their output straight up, untouched.
This keeps information fresh and prevents earlier knowledge from fading as the tower grows taller.

In other words, layers don't have to learn the *entire* transformation, they only need to learn the difference (the residual).

#### Step-by-Step Summary

| Step | Action     | Description                                  |
| ---- | ---------- | -------------------------------------------- |
| 1    | Input      | $x$ enters the block                         |
| 2    | Transform  | Apply operations to compute $\mathcal{F}(x)$ |
| 3    | Skip       | Pass $x$ directly to the output              |
| 4    | Combine    | Add: $y = \mathcal{F}(x) + x$                |
| 5    | Next block | Feed $y$ forward to the next layer           |

#### Example

In a simple feedforward block:
$$
\mathcal{F}(x) = W_2 \sigma(W_1 x)
$$
then:
$$
y = W_2 \sigma(W_1 x) + x
$$

If $\mathcal{F}(x)$ becomes very small (close to zero),
the block simply passes input forward: $y \approx x$, 
making it easy to train identity mappings when needed.

#### Tiny Code (Easy Versions)

Python (NumPy)

```python
import numpy as np

def relu(x): return np.maximum(0, x)

def residual_block(x, W1, W2):
    F = relu(x @ W1)
    F = F @ W2
    return F + x  # skip connection
```

C (Outline)

```c
void residual_block(double* x, double* W1, double* W2, double* y, int n, int m) {
    // Compute F(x)
    double* F = malloc(sizeof(double) * n);
    for (int i = 0; i < n; i++) {
        F[i] = 0.0;
        for (int j = 0; j < m; j++)
            F[i] += x[j] * W1[j * n + i];
        if (F[i] < 0) F[i] = 0; // ReLU
    }
    // Second linear + skip connection
    for (int i = 0; i < n; i++) {
        y[i] = F[i];
        y[i] += x[i]; // add skip
    }
    free(F);
}
```

#### Why It Matters

- Fixes vanishing gradients: gradients flow through identity path.
- Enables ultra-deep networks: ResNets up to 1000+ layers train reliably.
- Improves generalization: encourages small, incremental refinements.
- Simplifies learning: layers learn *residual corrections*, not full mappings.
- Ubiquitous in modern architectures: ResNet, DenseNet, Transformers.

#### A Gentle Proof (Why It Works)

During backpropagation, the gradient flows through both paths:

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \left( \frac{\partial \mathcal{F}(x)}{\partial x} + I \right)
$$

The identity term ($I$) ensures that the gradient never becomes zero, even if $\frac{\partial \mathcal{F}(x)}{\partial x}$ vanishes, maintaining stable gradient flow across deep stacks.

#### Variants

1. Pre-activation ResNet
   Apply BatchNorm and ReLU *before* $\mathcal{F}(x)$ for smoother training.
   $$
   y = x + \mathcal{F}(\text{BN}(\text{ReLU}(x)))
   $$

2. Projection shortcut (for dimension mismatch)
   If $\mathcal{F}(x)$ changes dimensions:
   $$
   y = \mathcal{F}(x) + W_s x
   $$
   where $W_s$ is a $1 \times 1$ convolution or linear projection.

#### Try It Yourself

1. Train a 10-layer MLP with and without skip connections, note which converges faster.
2. Visualize gradient norms per layer.
3. Try stacking 100+ residual blocks.
4. Replace addition with concatenation (DenseNet-style).
5. Combine with normalization and activation ordering (pre-activation).

#### Test Cases

| Architecture  | Residuals        | Result              |
| ------------- | ---------------- | ------------------- |
| 10-layer MLP  | None             | Vanishing gradients |
| 10-layer MLP  | Skip connections | Stable              |
| 100-layer CNN | None             | Diverges            |
| 100-layer CNN | Residual         | Trains successfully |

#### Complexity

- Time: negligible overhead (just addition)
- Memory: adds one buffer for skip tensor
- Effect: stabilizes training of deep models

Residual connections turned depth from a problem into a superpower, they let networks grow taller, learn faster, and see farther without forgetting where they came from.

# Section 95. Sequence Models 

### 941. Hidden Markov Model (Forward–Backward Algorithm)

The Hidden Markov Model (HMM) is a foundational probabilistic model for sequential data, where we observe outcomes that depend on hidden (unseen) states. The Forward–Backward algorithm allows us to efficiently compute probabilities of sequences and estimate hidden states over time.

#### What Problem Are We Solving?

Suppose we observe a sequence of outputs (like words, sounds, or sensor readings), but we suspect these were generated by hidden internal states, for example:

- In speech recognition: hidden phonemes → observed sounds
- In finance: hidden market regimes → observed prices
- In biology: hidden gene states → observed nucleotide sequences

We want to compute:

1. The likelihood of the observed sequence given the model parameters.
2. The posterior probabilities of hidden states over time.

That's what the Forward–Backward algorithm does.

#### The Core Idea

We define:

- Hidden states: $S = {s_1, s_2, \dots, s_N}$
- Observations: $O = (o_1, o_2, \dots, o_T)$
- Transition probabilities: $A_{ij} = P(s_j \mid s_i)$
- Emission probabilities: $B_j(o_t) = P(o_t \mid s_j)$
- Initial probabilities: $\pi_i = P(s_i \text{ at } t=1)$

We want $P(O \mid \text{model})$.

#### Step 1: Forward Pass (α values)

Compute probability of partial observation up to time $t$ and ending in state $i$:

$$
\alpha_t(i) = P(o_1, o_2, \dots, o_t, s_i \mid \text{model})
$$

Recurrence:

$$
\alpha_t(i) = \left[\sum_{j=1}^N \alpha_{t-1}(j) A_{ji}\right] B_i(o_t)
$$

Initialize:

$$
\alpha_1(i) = \pi_i B_i(o_1)
$$

#### Step 2: Backward Pass (β values)

Compute probability of the remaining observations from $t+1$ to $T$, given state $i$ at time $t$:

$$
\beta_t(i) = P(o_{t+1}, \dots, o_T \mid s_i, \text{model})
$$

Recurrence:

$$
\beta_t(i) = \sum_{j=1}^N A_{ij} B_j(o_{t+1}) \beta_{t+1}(j)
$$

Initialize:

$$
\beta_T(i) = 1
$$

#### Step 3: Combine (Posterior Probabilities)

The probability of being in state $i$ at time $t$:

$$
\gamma_t(i) = \frac{\alpha_t(i) \beta_t(i)}{\sum_{k=1}^N \alpha_t(k) \beta_t(k)}
$$

The total sequence likelihood:

$$
P(O \mid \text{model}) = \sum_{i=1}^N \alpha_T(i)
$$

#### How It Works (Plain Language)

The forward pass collects evidence from the past, "How likely are we to reach this state given everything so far?"
The backward pass collects evidence from the future, "How likely are we to see the rest of the data given we are here now?"

By multiplying them, we see the full picture: the probability that each hidden state was active at each time step.

#### Tiny Code (Easy Versions)

Python

```python
import numpy as np

def forward_backward(A, B, pi, O):
    N = A.shape[0]
    T = len(O)

    # Forward
    alpha = np.zeros((T, N))
    alpha[0] = pi * B[:, O[0]]
    for t in range(1, T):
        alpha[t] = (alpha[t-1] @ A) * B[:, O[t]]

    # Backward
    beta = np.zeros((T, N))
    beta[-1] = 1
    for t in reversed(range(T-1)):
        beta[t] = (A @ (B[:, O[t+1]] * beta[t+1]))

    # Posterior
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    return alpha, beta, gamma
```

C (Outline)

```c
for (int i = 0; i < N; i++)
    alpha[0][i] = pi[i] * B[i][O[0]];

for (int t = 1; t < T; t++)
    for (int i = 0; i < N; i++) {
        alpha[t][i] = 0;
        for (int j = 0; j < N; j++)
            alpha[t][i] += alpha[t-1][j] * A[j][i];
        alpha[t][i] *= B[i][O[t]];
    }
```

#### Why It Matters

- Foundation of probabilistic sequence models, used in speech, NLP, and bioinformatics.
- Efficient dynamic programming, avoids exponential complexity.
- Forms the basis of EM training (Baum–Welch algorithm).
- Interpretable, gives posterior state probabilities over time.

Without it, computing probabilities across all possible state paths would take $O(N^T)$ time; this reduces it to $O(N^2T)$.

#### A Gentle Proof (Why It Works)

Because the model is Markovian, the probability of reaching a state depends only on the previous state, not the entire history.
This allows recursive computation of partial probabilities (forward) and remaining probabilities (backward).

The product $\alpha_t(i)\beta_t(i)$ covers both halves of the sequence, before and after time $t$, giving the full joint probability.

#### Try It Yourself

1. Build a simple weather model (Sunny/Rainy).
2. Define $A$, $B$, $\pi$, and observation sequence.
3. Run the forward–backward algorithm.
4. Plot $\gamma_t(i)$ for each state, see how hidden states fluctuate.
5. Compare with known labels to verify interpretation.

#### Test Cases

| Example | States        | Observations       | Goal                        |
| ------- | ------------- | ------------------ | --------------------------- |
| Weather | Rainy, Sunny  | Walk, Shop, Clean  | Infer most likely weather   |
| Speech  | Phonemes      | Audio features     | Compute state probabilities |
| DNA     | Hidden motifs | Nucleotide symbols | Find likely motif positions |

#### Complexity

- Time: $O(N^2T)$
- Space: $O(NT)$
- Benefit: Efficient inference in hidden state models

The Forward–Backward algorithm is how we "listen" to hidden processes, blending past and future evidence to decode what we can't directly see.

### 942. Viterbi Algorithm

The Viterbi algorithm is a dynamic programming method used to find the most probable sequence of hidden states in a Hidden Markov Model (HMM), given a sequence of observations.
It is the backbone of modern speech recognition, part-of-speech tagging, gene sequence decoding, and many other sequence labeling tasks.

#### What Problem Are We Solving?

In an HMM, many different state sequences could produce the same observations.
We want to find the single most likely path of hidden states:

$$
S^* = \arg\max_S P(S \mid O, \text{model})
$$

Naively, this requires checking all possible paths, an exponential number in $T$.
The Viterbi algorithm makes this efficient using dynamic programming.

#### The Core Idea

We recursively compute the maximum probability of any path ending in a given state at each time step.

Define:

- $A_{ij} = P(s_j \mid s_i)$ (state transition)
- $B_j(o_t) = P(o_t \mid s_j)$ (emission)
- $\pi_i = P(s_i)$ (initial state)
- Observations $O = (o_1, o_2, \dots, o_T)$

We define:

$$
\delta_t(i) = \max_{s_1, \dots, s_{t-1}} P(s_1, \dots, s_t = i, o_1, \dots, o_t \mid \text{model})
$$

and a backpointer $\psi_t(i)$ to remember which state led to the best path.

#### Step-by-Step Algorithm

1. Initialization

$$
\delta_1(i) = \pi_i B_i(o_1), \quad \psi_1(i) = 0
$$

2. Recursion

For each time $t = 2, \dots, T$:

$$
\delta_t(i) = \max_j [\delta_{t-1}(j) A_{ji}] , B_i(o_t)
$$

and record the argmax:

$$
\psi_t(i) = \arg\max_j [\delta_{t-1}(j) A_{ji}]
$$

3. Termination

$$
P^* = \max_i \delta_T(i), \quad s_T^* = \arg\max_i \delta_T(i)
$$

4. Path Backtracking

For $t = T-1, T-2, \dots, 1$:

$$
s_t^* = \psi_{t+1}(s_{t+1}^*)
$$

#### How It Works (Plain Language)

Think of it like navigating a maze where each junction represents a hidden state and each step an observation.
At each step, you only keep the best path that led to each possible state, discarding all others.
By the end, you just walk backward through the saved pointers to recover the most likely route.

#### Example

Let's say we have two hidden states:
Rainy (R) and Sunny (S), and observed actions: Walk, Shop, Clean.

We can compute the most likely weather sequence that explains these actions using the Viterbi algorithm, just like teaching a model to "guess the sky" from someone's habits.

#### Tiny Code (Easy Versions)

Python

```python
import numpy as np

def viterbi(A, B, pi, O):
    N = A.shape[0]
    T = len(O)
    delta = np.zeros((T, N))
    psi = np.zeros((T, N), dtype=int)

    # Initialization
    delta[0] = pi * B[:, O[0]]

    # Recursion
    for t in range(1, T):
        for i in range(N):
            seq_probs = delta[t-1] * A[:, i]
            psi[t, i] = np.argmax(seq_probs)
            delta[t, i] = np.max(seq_probs) * B[i, O[t]]

    # Termination
    states = np.zeros(T, dtype=int)
    states[-1] = np.argmax(delta[-1])
    for t in reversed(range(T-1)):
        states[t] = psi[t+1, states[t+1]]

    return states
```

C (Outline)

```c
for (int i = 0; i < N; i++)
    delta[0][i] = pi[i] * B[i][O[0]];

for (int t = 1; t < T; t++)
    for (int i = 0; i < N; i++) {
        double max_val = 0.0;
        int argmax = 0;
        for (int j = 0; j < N; j++) {
            double val = delta[t-1][j] * A[j][i];
            if (val > max_val) { max_val = val; argmax = j; }
        }
        delta[t][i] = max_val * B[i][O[t]];
        psi[t][i] = argmax;
    }
```

#### Why It Matters

- Efficient decoding, reduces exponential search to $O(N^2T)$.
- Most probable state sequence, not just per-state probabilities.
- Used everywhere, speech, POS tagging, gesture recognition, DNA sequencing.
- Foundation for structured prediction in sequence models.

#### A Gentle Proof (Why It Works)

We use dynamic programming:
At each time $t$, the optimal path to state $i$ must come from the optimal path to some previous state $j$ at time $t-1$.

Formally, the Bellman optimality principle ensures:

$$
\max_{s_1,\dots,s_t} P(s_1,\dots,s_t,o_1,\dots,o_t) =
\max_j [\max_{s_1,\dots,s_{t-1}} P(s_1,\dots,s_{t-1},o_1,\dots,o_{t-1}) P(s_t=i \mid s_{t-1}=j) P(o_t \mid s_t)]
$$

This recursion allows efficient accumulation of probabilities.

#### Try It Yourself

1. Define a 2-state (Rainy, Sunny) HMM with 3 observations (Walk, Shop, Clean).
2. Run the Viterbi algorithm manually or in Python.
3. Track $\delta_t(i)$ at each step in a table.
4. Recover the best state path and compare with your intuition.
5. Visualize the path probabilities, see where the decision flips.

#### Test Cases

| Example         | States          | Observations      | Output                       |
| --------------- | --------------- | ----------------- | ---------------------------- |
| Weather model   | Rainy, Sunny    | Walk, Shop, Clean | [Rainy, Rainy, Sunny]        |
| Speech phonemes | Hidden phonemes | Acoustic signals  | Most likely phoneme sequence |
| DNA decoding    | Hidden regions  | Base pairs        | Gene region sequence         |

#### Complexity

- Time: $O(N^2T)$
- Space: $O(NT)$
- Output: single most probable hidden state sequence

The Viterbi algorithm turns uncertainty into structure, instead of guessing, it reconstructs the hidden path that best explains the story behind every observed sequence.

### 943. Baum–Welch Algorithm

The Baum–Welch algorithm is the classic way to train a Hidden Markov Model (HMM) when the hidden states are unknown.
It is an instance of the Expectation–Maximization (EM) algorithm, alternating between estimating expected counts of transitions and emissions, and updating model parameters accordingly.

It allows an HMM to *learn from data*, discovering transition probabilities, emission probabilities, and initial distributions that best explain the observed sequences.

#### What Problem Are We Solving?

Suppose we have a dataset of observed sequences $O = (o_1, o_2, \dots, o_T)$ but no knowledge of which hidden states produced them.
We want to learn the HMM parameters $\lambda = (A, B, \pi)$ that maximize the likelihood:

$$
P(O \mid \lambda)
$$

Directly optimizing this is infeasible, there are exponentially many hidden state sequences.
Baum–Welch solves this by using expected counts of transitions and emissions, computed via the Forward–Backward algorithm.

#### The Core Idea

Each iteration has two steps:

1. Expectation (E-step)
   Compute the expected number of times each transition or emission occurred, given the current model.
2. Maximization (M-step)
   Re-estimate the model parameters $A, B, \pi$ using those expected counts.

Repeat until convergence.

#### Step 1: Compute Probabilities

Using Forward–Backward from before, compute:

- Forward probability:

  $$
  \alpha_t(i) = P(o_1, \dots, o_t, s_t = i \mid \lambda)
  $$

- Backward probability:

  $$
  \beta_t(i) = P(o_{t+1}, \dots, o_T \mid s_t = i, \lambda)
  $$

Then define two expected quantities:

1. State occupancy probability:

   $$
   \gamma_t(i) = \frac{\alpha_t(i)\beta_t(i)}{P(O \mid \lambda)}
   $$

2. Transition probability:

   $$
   \xi_t(i,j) = \frac{\alpha_t(i) A_{ij} B_j(o_{t+1}) \beta_{t+1}(j)}{P(O \mid \lambda)}
   $$

#### Step 2: Re-estimation Formulas

With $\gamma_t(i)$ and $\xi_t(i,j)$ computed, update:

- Initial state distribution:

  $$
  \pi_i' = \gamma_1(i)
  $$

- Transition matrix:

  $$
  A_{ij}' = \frac{\sum_{t=1}^{T-1} \xi_t(i,j)}{\sum_{t=1}^{T-1} \gamma_t(i)}
  $$

- Emission probabilities:

  $$
  B_j'(k) = \frac{\sum_{t: o_t = k} \gamma_t(j)}{\sum_{t=1}^T \gamma_t(j)}
  $$

Repeat until $P(O \mid \lambda)$ converges (or increases by less than a small $\epsilon$).

#### How It Works (Plain Language)

Think of Baum–Welch as guessing, explaining, and adjusting:

1. Guess the model parameters.
2. Explain the data, use Forward–Backward to estimate which hidden states likely generated each observation.
3. Adjust, reassign probability mass to transitions and emissions that better match those hidden-state expectations.

Each cycle refines the model, making the explanation more plausible.

#### Example

For a weather model:

- Hidden states: Rainy, Sunny
- Observations: Walk, Shop, Clean

If the algorithm sees many "Walk" observations, it gradually increases $P(\text{Walk} \mid \text{Sunny})$.
If "Clean" often follows "Walk," it adjusts transition $P(\text{Sunny} \to \text{Rainy})$.
This iterative learning continues until the model stabilizes.

#### Tiny Code (Simplified)

Python

```python
import numpy as np

def baum_welch(O, N, M, max_iters=100):
    T = len(O)
    A = np.full((N, N), 1.0 / N)
    B = np.full((N, M), 1.0 / M)
    pi = np.full(N, 1.0 / N)

    for _ in range(max_iters):
        alpha = np.zeros((T, N))
        beta = np.zeros((T, N))
        # Forward
        alpha[0] = pi * B[:, O[0]]
        for t in range(1, T):
            alpha[t] = (alpha[t-1] @ A) * B[:, O[t]]
        # Backward
        beta[-1] = 1
        for t in reversed(range(T-1)):
            beta[t] = (A @ (B[:, O[t+1]] * beta[t+1]))
        # Gamma & Xi
        denom = (alpha[-1].sum())
        gamma = (alpha * beta) / denom
        xi = np.zeros((T-1, N, N))
        for t in range(T-1):
            xi[t] = (alpha[t][:, None] * A * B[:, O[t+1]] * beta[t+1]) / denom
        # Re-estimate
        pi = gamma[0]
        A = xi.sum(axis=0) / gamma[:-1].sum(axis=0)[:, None]
        for j in range(N):
            for k in range(M):
                mask = np.array(O) == k
                B[j, k] = gamma[mask, j].sum() / gamma[:, j].sum()
    return A, B, pi
```

#### Why It Matters

- Teaches HMMs to learn from data, without labeled states.
- Foundation for speech recognition, NLP tagging, and biological sequence modeling.
- Unsupervised EM framework, interpretable and statistically sound.
- Used before neural sequence models dominated modern AI.

Even today, it remains a conceptual foundation for EM and latent-variable models.

#### A Gentle Proof (Why It Works)

Baum–Welch maximizes a lower bound on $\log P(O \mid \lambda)$ using the EM principle.

- E-step: compute expected sufficient statistics of hidden variables given $\lambda^{(old)}$.
- M-step: maximize expected log-likelihood over $\lambda$.

Each iteration guarantees:

$$
P(O \mid \lambda^{(new)}) \ge P(O \mid \lambda^{(old)})
$$

This monotonic improvement continues until convergence.

#### Try It Yourself

1. Generate a synthetic sequence from a known HMM.
2. Erase the hidden states.
3. Initialize random $A$, $B$, $\pi$.
4. Run Baum–Welch for several iterations.
5. Compare learned parameters with the true ones.

You'll see the model "rediscover" the hidden dynamics from the observations alone.

#### Test Cases

| Example     | Hidden States   | Observations      | Goal                                     |
| ----------- | --------------- | ----------------- | ---------------------------------------- |
| Weather     | Rainy, Sunny    | Walk, Shop, Clean | Learn $A$, $B$, $\pi$                    |
| DNA         | Gene/Non-Gene   | Nucleotides       | Estimate transition + emission structure |
| POS Tagging | Noun, Verb, Adj | Words             | Learn emission patterns                  |

#### Complexity

- Time: $O(N^2T)$ per iteration
- Space: $O(NT)$
- Convergence: monotonic but not guaranteed to global optimum

Baum–Welch is the "teacher" of Hidden Markov Models, it listens to sequences, guesses the unseen causes, and refines its understanding until the hidden structure of the world becomes clear.

### 944. Beam Search

Beam Search is a heuristic search algorithm widely used in sequence decoding, especially in speech recognition, neural machine translation, and text generation.
It is a clever compromise between greedy search (fast but myopic) and exhaustive search (accurate but expensive), balancing efficiency and quality by exploring only the most promising candidate paths at each step.

#### What Problem Are We Solving?

Many sequence models (like RNNs, Transformers, or HMMs) generate outputs step by step, predicting the next symbol based on prior ones.
The number of possible sequences grows exponentially with length, making exact decoding infeasible.

We want to find a high-probability output sequence without exploring every possible path.
Beam Search achieves this by keeping only the top $k$ candidates (the "beam") at each step.

#### The Core Idea

Let the model define conditional probabilities:

$$
P(y_1, y_2, \dots, y_T) = \prod_{t=1}^{T} P(y_t \mid y_{<t})
$$

At each decoding step $t$, instead of taking only the single most likely token (greedy), or all tokens (exhaustive), Beam Search keeps the $k$ most probable partial sequences according to their cumulative log probability.

Formally, define:

- Beam width $k$ (e.g., 3–10)
- Cumulative log probability:

$$
\text{score}(y_{1:t}) = \sum_{i=1}^t \log P(y_i \mid y_{<i})
$$

At each step, expand all candidates by one token, compute new scores, and keep only the best $k$.

#### Step-by-Step Summary

| Step | Description                                                     |
| ---- | --------------------------------------------------------------- |
| 1    | Start with initial token (e.g., `<start>`) and empty beam       |
| 2    | Expand each beam by all possible next tokens                    |
| 3    | Compute log probabilities for all expanded sequences            |
| 4    | Select top $k$ sequences with highest scores                    |
| 5    | Repeat until end-of-sequence `<eos>` is reached or length limit |
| 6    | Return the sequence with highest score (or normalized score)    |

#### How It Works (Plain Language)

Imagine you're guessing a sentence word by word.
At each step, you write down the few most likely sentences so far, say the top 3.
Then you extend *each* of those by one new word, compute their new probabilities, and again keep only the best 3 overall.
You keep going until the sentence ends.

Beam Search doesn't guarantee the absolute best sentence, but it almost always finds a very good one, much faster than trying them all.

#### Example

Suppose your model can generate only `A`, `B`, or `C` at each step, and beam width $k=2$.

At step 1:

| Candidate | Prob |
| --------- | ---- |
| A         | 0.6  |
| B         | 0.3  |
| C         | 0.1  |

Keep top 2 → {A, B}

At step 2 (expanding A and B):

| Sequence | Cumulative Prob |
| -------- | --------------- |
| AA       | 0.6×0.5=0.30    |
| AB       | 0.6×0.3=0.18    |
| BA       | 0.3×0.4=0.12    |
| BB       | 0.3×0.6=0.18    |

Keep top 2 → {AA, AB}
Continue until end.

#### Tiny Code (Easy Versions)

Python

```python
import numpy as np

def beam_search_step(probs, beam, k):
    new_beam = []
    for seq, score in beam:
        for token, p in enumerate(probs):
            new_beam.append((seq + [token], score + np.log(p)))
    # Keep top k
    new_beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:k]
    return new_beam

# Example
probs_t1 = [0.6, 0.3, 0.1]
beam = [([], 0)]
beam = beam_search_step(probs_t1, beam, 2)
print(beam)
```

C (Outline)

```c
typedef struct {
    int seq[100];
    double score;
} Beam;

void beam_search_step(double *probs, Beam *beam, Beam *next, int k, int vocab_size) {
    int count = 0;
    for (int i = 0; i < k; i++)
        for (int t = 0; t < vocab_size; t++) {
            next[count] = beam[i];
            next[count].seq[i+1] = t;
            next[count].score += log(probs[t]);
            count++;
        }
    // sort and prune to top k (omitted for brevity)
}
```

#### Why It Matters

- Balances accuracy and efficiency, between greedy and exhaustive search.
- Standard in decoding for neural sequence models.
- Allows exploration of multiple hypotheses.
- Often used with normalization tricks (e.g., length penalty).
- Simple, general, and tunable.

Beam width controls the trade-off:

- Small $k$ → fast but sometimes suboptimal.
- Large $k$ → better results but slower decoding.

#### A Gentle Proof (Why It Works)

Beam Search is not optimal (it prunes), but it's a heuristic approximation to best-first search.
By keeping top-$k$ partial hypotheses under cumulative probability, it approximates:

$$
S^* = \arg\max_S \prod_t P(y_t \mid y_{<t})
$$

Because log is monotonic, maximizing the log sum yields the same order as maximizing product probabilities.

#### Try It Yourself

1. Train a simple RNN for text generation.
2. Compare three decoding modes:

   * Greedy ($k=1$)
   * Beam ($k=3$)
   * Sampling (stochastic decoding)
3. Observe the trade-off between coherence and diversity.
4. Add length normalization:
   $$
   \text{score}' = \frac{\text{score}}{(5 + |y|)^\alpha / (5 + 1)^\alpha}
   $$
   to prevent short sentences from dominating.

#### Test Cases

| Model       | Task              | Beam Width | Result                      |
| ----------- | ----------------- | ---------- | --------------------------- |
| RNN         | Text generation   | 1          | Coherent but dull           |
| RNN         | Text generation   | 5          | More fluent, longer context |
| Transformer | Translation       | 4–8        | State-of-the-art results    |
| HMM         | Sequence labeling | 3          | Near-optimal decoding       |

#### Complexity

- Time: $O(k \cdot T \cdot V)$
  (beam width × sequence length × vocabulary size)
- Space: $O(k \cdot T)$
- Trade-off: quality vs computation

Beam Search is the decoder's workhorse, pruning wisely, it finds the path that feels right, even if it's not the only one that could have been.

### 945. Greedy Decoding

Greedy Decoding is the simplest decoding method for sequence models.
At every time step, it chooses the most probable next token according to the model's current prediction, without reconsidering earlier choices.

It's fast, deterministic, and often surprisingly effective, though it can miss globally optimal or more diverse solutions.

#### What Problem Are We Solving?

Given a model that defines a probability over sequences:

$$
P(y_1, y_2, \dots, y_T) = \prod_{t=1}^{T} P(y_t \mid y_{<t})
$$

we want to find the output sequence with the highest probability.

Full search is exponential in $T$, so we approximate it.
Greedy decoding picks the best local choice at each step:

$$
y_t^* = \arg\max_y P(y \mid y_{<t})
$$

This yields a single sequence, quickly.

#### The Core Idea

Instead of exploring multiple candidates (like Beam Search), we take a single path through the sequence, always choosing the next token with the highest conditional probability.

Formally:

$$
\hat{y} = [,y_1^*, y_2^*, \dots, y_T^*,], \quad
y_t^* = \arg\max_y P(y \mid y_{<t})
$$

No backtracking, no search, just straightforward next-step prediction.

#### How It Works (Plain Language)

Imagine you're building a sentence word by word, always picking the word that "feels most right" at that moment.
You never look back to fix an earlier mistake, and you never explore alternative phrasings.

Greedy decoding is that, a single straight-line walk through the model's probability landscape.

#### Step-by-Step Summary

| Step | Action  | Description                                           |
| ---- | ------- | ----------------------------------------------------- |
| 1    | Start   | Begin with a special `<start>` token                  |
| 2    | Predict | Compute next-token probabilities $P(y_t \mid y_{<t})$ |
| 3    | Select  | Choose the token with the highest probability         |
| 4    | Append  | Add it to the output sequence                         |
| 5    | Repeat  | Continue until `<eos>` or length limit                |

#### Example

If your model outputs token probabilities like this:

| Step | Token Probabilities  | Chosen |
| ---- | -------------------- | ------ |
| 1    | {"A": 0.6, "B": 0.4} | "A"    |
| 2    | {"A": 0.3, "C": 0.7} | "C"    |
| 3    | {"B": 0.8, "D": 0.2} | "B"    |

Then the greedy-decoded output is: A C B.

It's fast, but if "A C B" leads to a dead end while "A B B" had higher total probability, greedy decoding won't find that.

#### Tiny Code (Easy Versions)

Python

```python
import numpy as np

def greedy_decode(model, start_token, max_len):
    seq = [start_token]
    for _ in range(max_len):
        probs = model.predict_next(seq)
        next_token = np.argmax(probs)
        seq.append(next_token)
        if next_token == model.eos_token:
            break
    return seq
```

C (Outline)

```c
int greedy_decode(double (*predict_next)(int*, int), int start, int eos, int max_len, int *output) {
    output[0] = start;
    for (int t = 1; t < max_len; t++) {
        int next = argmax(predict_next(output, t));
        output[t] = next;
        if (next == eos) break;
    }
    return 0;
}
```

#### Why It Matters

- Fast and simple: no beam management or probability bookkeeping.
- Deterministic: same input → same output.
- Useful for debugging or baseline generation.
- Strong for confident models: works well when probabilities are sharply peaked.
- Weak for ambiguous or uncertain models: may miss better long-term paths.

Greedy decoding is widely used for inference in models that are well-calibrated, like some classification or captioning systems.

#### A Gentle Proof (Why It Works and When It Doesn't)

Let's define the global optimum as:

$$
y^* = \arg\max_y P(y)
$$

and the greedy sequence as:

$$
\hat{y}*t = \arg\max_y P(y_t \mid \hat{y}*{<t})
$$

Greedy decoding maximizes *locally* at each step, not *globally*.
By the inequality:

$$
\max_y \prod_t P(y_t \mid y_{<t}) \neq \prod_t \max_y P(y_t \mid y_{<t})
$$

it can fail to reach the global maximum.
However, if $P(y_t \mid y_{<t})$ is sharply peaked (low entropy), greedy often approximates the true best sequence.

#### Try It Yourself

1. Train a small character-level language model.
2. Decode text using:

   * Greedy decoding
   * Beam Search ($k=3$)
   * Sampling (randomized)
3. Compare output quality and diversity.
4. Observe: Greedy outputs are fluent but repetitive; beam outputs are more coherent; sampling is more creative.

#### Test Cases

| Model        | Task              | Decoding | Result                          |
| ------------ | ----------------- | -------- | ------------------------------- |
| Seq2Seq      | Translation       | Greedy   | Fast, sometimes short sentences |
| Transformer  | Text generation   | Greedy   | Fluent but deterministic        |
| Speech model | Transcription     | Greedy   | Works well when signal clear    |
| RNN          | Poetry generation | Greedy   | Repetitive phrases              |

#### Complexity

- Time: $O(T \cdot V)$ (same as model inference)
- Space: $O(T)$
- Trade-off: speed vs exploration

Greedy decoding is the quick and confident storyteller, it never doubts its next word, even if a little hesitation might have led to something greater.

### 946. Connectionist Temporal Classification (CTC)

Connectionist Temporal Classification (CTC) is a training and decoding method designed for sequence-to-sequence tasks where the alignment between inputs and outputs is unknown, like speech recognition, handwriting recognition, or gesture decoding.

It allows neural networks (especially RNNs or Transformers) to map variable-length input sequences to shorter output sequences without explicit frame-level alignment.

#### What Problem Are We Solving?

Suppose we have:

- Input sequence $X = (x_1, x_2, \dots, x_T)$ (like audio frames)
- Target output sequence $Y = (y_1, y_2, \dots, y_L)$ (like letters or words)

We don't know *which input frame corresponds to which output token*.
Traditional supervised learning requires alignment, but labeling every frame is expensive.
CTC solves this by marginalizing over all possible alignments between inputs and outputs.

#### The Core Idea

CTC introduces an extra "blank" symbol (∅) and allows repetitions of labels.
A valid alignment $\pi = (\pi_1, \pi_2, \dots, \pi_T)$ maps to a final output $Y$ by:

1. Removing all blanks (∅).
2. Collapsing repeated symbols.

Example:
$\pi = [∅, h, h, ∅, e, e, ∅, l, l, o, ∅]$ → $Y = [h, e, l, o]$

The model predicts a probability distribution over all possible alignments.

The total probability of output $Y$ is:

$$
P(Y \mid X) = \sum_{\pi \in \mathcal{B}^{-1}(Y)} P(\pi \mid X)
$$

where $\mathcal{B}$ is the collapsing function (remove blanks + merge repeats).

#### Step-by-Step Summary

| Step | Description                                                       |
| ---- | ----------------------------------------------------------------- |
| 1    | Define vocabulary with blank (∅) symbol                           |
| 2    | Network outputs probabilities $P(\pi_t \mid x_t)$ per frame       |
| 3    | Enumerate all alignments $\pi$ that map to $Y$                    |
| 4    | Sum probabilities of all valid paths (Forward–Backward algorithm) |
| 5    | Maximize $\log P(Y \mid X)$ during training                       |

#### The CTC Objective

For a target sequence $Y = (y_1, \dots, y_L)$, define an extended sequence by inserting blanks between symbols and at the ends:

$$
\tilde{Y} = (∅, y_1, ∅, y_2, ∅, \dots, y_L, ∅)
$$

Then recursively compute forward probabilities $\alpha_t(s)$ over positions $s$ in $\tilde{Y}$:

Initialization:
$$
\alpha_1(1) = P(∅ \mid x_1), \quad \alpha_1(2) = P(y_1 \mid x_1)
$$

Recurrence:
$$
\alpha_t(s) = P(\tilde{Y}*s \mid x_t) \times \left[
\alpha*{t-1}(s) + \alpha_{t-1}(s-1) + \mathbb{1}*{\tilde{Y}*s \ne \tilde{Y}*{s-2}} \alpha*{t-1}(s-2)
\right]
$$

Final probability:
$$
P(Y \mid X) = \alpha_T(S-1) + \alpha_T(S)
$$
where $S = |\tilde{Y}|$.

#### How It Works (Plain Language)

Imagine listening to a sentence: some sounds are stretched, some skipped, some blurred.
CTC tells the network:

> "Don't worry about timing. Just make sure that, somewhere in your predictions, the right sequence of symbols appears, in order."

The blank symbol allows pauses and variable timing; summing over alignments lets the model learn flexible mappings automatically.

#### Example

If the input has 5 frames and the target is "HI", the possible alignments include:

| Frame Sequence  | Collapsed Output |
| --------------- | ---------------- |
| [H, H, ∅, I, I] | HI               |
| [∅, H, ∅, I, ∅] | HI               |
| [H, ∅, H, I, ∅] | HI               |

CTC sums over all of them, not just one.

#### Tiny Code (Simplified)

Python (Pseudo)

```python
import numpy as np

def ctc_forward(log_probs, target, blank=0):
    T, V = log_probs.shape
    target_ext = [blank] + [t for t in target for _ in (0, 1)] + [blank]
    S = len(target_ext)
    alpha = np.full((T, S), -np.inf)
    alpha[0, 0] = log_probs[0, blank]
    alpha[0, 1] = log_probs[0, target_ext[1]]

    for t in range(1, T):
        for s in range(S):
            prev = [alpha[t-1, s]]
            if s > 0: prev.append(alpha[t-1, s-1])
            if s > 1 and target_ext[s] != target_ext[s-2]:
                prev.append(alpha[t-1, s-2])
            alpha[t, s] = np.logaddexp.reduce(prev) + log_probs[t, target_ext[s]]
    return np.logaddexp(alpha[T-1, S-1], alpha[T-1, S-2])
```

C (Outline)

```c
// compute alpha[t][s] forward matrix (log-sum-exp form)
// requires careful numerical stability handling
```

#### Why It Matters

- No need for pre-aligned data, learns timing automatically.
- Robust to variable-length inputs and outputs.
- Foundation of speech recognition models like DeepSpeech.
- Still used in modern architectures (e.g., wav2vec 2.0 pretraining).
- Differentiable and trainable end-to-end.

Without CTC, many sequence problems (like speech or handwriting) would require labor-intensive frame-level labeling.

#### A Gentle Proof (Why It Works)

Because each output symbol can appear in multiple time frames, we marginalize over alignments.
Using the Forward–Backward algorithm, we efficiently sum the exponentially many paths:

$$
P(Y \mid X) = \sum_{\pi \in \mathcal{B}^{-1}(Y)} \prod_{t=1}^T P(\pi_t \mid x_t)
$$

This can be computed recursively in $O(TL)$ time, not $O(V^T)$.

#### Try It Yourself

1. Generate a toy input of 5 timesteps and target word "CAT".
2. Enumerate all possible alignments.
3. Collapse them using the blank rule.
4. Compute probabilities manually and verify with CTC forward recursion.
5. Try training a small RNN to map sine-wave patterns to letter sequences.

#### Test Cases

| Application             | Input        | Output  | Alignment Known? |
| ----------------------- | ------------ | ------- | ---------------- |
| Speech recognition      | Audio frames | Text    | No               |
| Handwriting recognition | Pen strokes  | Text    | No               |
| Sign language           | Video frames | Glosses | No               |

#### Complexity

- Time: $O(TL)$
- Space: $O(TL)$
- Key advantage: Differentiable marginalization over alignments

CTC lets neural networks learn *what* to say without being told *when* to say it, transforming chaotic sequences into structured meaning, one blank at a time.

### 947. Attention Mechanism

The Attention Mechanism is one of the most transformative ideas in modern AI.
It allows a model to focus selectively on the most relevant parts of its input when making a decision, just as humans do when reading, listening, or reasoning.

Originally introduced for neural machine translation, attention is now a core component of Transformers, powering GPT, BERT, and nearly every state-of-the-art model in language, vision, and beyond.

#### What Problem Are We Solving?

Sequence models like RNNs compress all past information into a single hidden vector.
This makes it hard for them to remember long contexts, they forget earlier words or events.

We want a model that can:

- Look back over all previous positions,
- Weigh them by relevance,
- Combine them dynamically at each step.

That's exactly what attention does.

#### The Core Idea

At each decoding step $t$, the model computes how strongly it should attend to each encoder state $h_i$ (where $i = 1, 2, \dots, T$).

We define:

- Query vector $q_t$ (from the decoder state)
- Key vectors $k_i$ (from encoder states)
- Value vectors $v_i$ (information to extract)

The attention weights $\alpha_{t,i}$ measure the relevance of $h_i$ to $q_t$:

$$
e_{t,i} = \text{score}(q_t, k_i)
$$

Then normalize via softmax:

$$
\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_j \exp(e_{t,j})}
$$

Finally, compute the context vector as the weighted average:

$$
c_t = \sum_i \alpha_{t,i} v_i
$$

The decoder uses $c_t$ as additional information to predict the next token.

#### Common Scoring Functions

| Type                | Formula                                       | Notes                 |
| ------------------- | --------------------------------------------- | --------------------- |
| Dot                 | $e_{t,i} = q_t^\top k_i$                      | Simple, fast          |
| Scaled Dot          | $e_{t,i} = \frac{q_t^\top k_i}{\sqrt{d_k}}$   | Used in Transformers  |
| Additive (Bahdanau) | $e_{t,i} = v_a^\top \tanh(W_q q_t + W_k k_i)$ | Learnable combination |

#### How It Works (Plain Language)

Imagine translating a sentence word by word.
When translating "bank," the model should look closely at the nearby words, "river" or "loan", to decide its meaning.
Instead of relying on a single memory, attention lets the model look back at *all* words, weigh them, and use that context for its current decision.

Attention turns sequence modeling from a linear memory into a differentiable lookup table, the model "consults" all previous information on demand.

#### Step-by-Step Summary

| Step | Description                                                         |
| ---- | ------------------------------------------------------------------- |
| 1    | Compute similarity $e_{t,i}$ between query $q_t$ and each key $k_i$ |
| 2    | Apply softmax to get attention weights $\alpha_{t,i}$               |
| 3    | Compute weighted sum $c_t = \sum_i \alpha_{t,i} v_i$                |
| 4    | Combine $c_t$ with decoder state to generate next output            |
| 5    | Repeat for each decoding step                                       |

#### Example

For a simple translation model:
Input sentence: "Le chat dort."
Output so far: "The cat ..."
When predicting "sleeps," the attention weights peak at "dort," allowing the model to pull the right context.

| Source | Attention Weight |
| ------ | ---------------- |
| Le     | 0.05             |
| chat   | 0.20             |
| dort   | 0.75             |

#### Tiny Code (Easy Versions)

Python (Scaled Dot-Product Attention)

```python
import numpy as np

def attention(Q, K, V):
    d_k = K.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    weights = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)
    return weights @ V, weights
```

C (Outline)

```c
void attention(double *Q, double *K, double *V, double *out, int n, int d) {
    double scale = 1.0 / sqrt((double)d);
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < n; j++) {
            double score = 0;
            for (int k = 0; k < d; k++)
                score += Q[i*d + k] * K[j*d + k];
            score *= scale;
            double expv = exp(score);
            sum += expv;
            out[i*d + k] += (expv / sum) * V[j*d + k];
        }
    }
}
```

#### Why It Matters

- Solves long-term dependency problems in RNNs.
- Fully differentiable, trained end-to-end.
- Interpretable, attention weights visualize what the model "looks at."
- Forms the heart of the Transformer architecture (via multi-head self-attention).
- General mechanism, works for language, vision, time series, reinforcement learning, and more.

#### A Gentle Proof (Why It Works)

The context vector $c_t$ is the expected value of encoder representations under the attention distribution $\alpha_t$:

$$
c_t = \mathbb{E}_{i \sim \alpha_t}[v_i]
$$

Since $\alpha_t$ sums to 1 and depends smoothly on $q_t$, the entire mechanism is differentiable and trainable via backpropagation.
It learns to assign high weights to inputs that reduce loss, aligning relevant features dynamically.

#### Variants

1. Additive Attention (Bahdanau), nonlinear score function.
2. Multiplicative / Scaled Dot (Luong, Vaswani), efficient matrix multiplication form.
3. Self-Attention, queries, keys, and values come from the same sequence.
4. Multi-Head Attention, multiple subspaces capture different relations.

#### Try It Yourself

1. Implement dot-product attention on small random vectors.
2. Visualize attention weights for different queries.
3. Extend to self-attention by using the same matrix for Q, K, V.
4. Observe how multi-head attention mixes features.
5. Apply attention weights to words in a sentence, see which parts light up.

#### Test Cases

| Task                | Model           | Attention Type  | Result               |
| ------------------- | --------------- | --------------- | -------------------- |
| Machine translation | Seq2Seq         | Bahdanau        | Improves alignment   |
| Summarization       | Transformer     | Self-attention  | Captures context     |
| Vision              | ViT             | Multi-head      | Global pixel context |
| Speech recognition  | Transformer-ASR | Cross-attention | Stable decoding      |

#### Complexity

- Time: $O(T^2 d)$
- Space: $O(T^2)$
- Trade-off: better context vs quadratic cost

Attention is the bridge between memory and reasoning, letting neural networks not just remember, but *choose* what to remember.

### 948. Transformer Decoder

The Transformer Decoder is the engine of modern generative AI, a sequence model that uses self-attention instead of recurrence to process context.
It's the key component behind GPT-style models, capable of modeling long-range dependencies efficiently and producing coherent text, code, and more.

#### What Problem Are We Solving?

Traditional sequence models (RNNs, LSTMs) process inputs step by step.
This makes them slow, sequential, and limited in remembering far-away context.
Transformers replace recurrence with parallel self-attention, enabling global context access and massive scalability.

The decoder part specifically handles auto-regressive generation, predicting the next token given all previous ones.

#### The Core Idea

At each decoding step $t$, the model attends to:

1. All previous tokens in the output sequence (via *masked self-attention*).
2. All encoder outputs (via *encoder–decoder attention*).
3. Its own learned hidden representation, layer by layer.

Each decoder layer has three sub-layers:

1. Masked Self-Attention

   * Computes attention over past positions only (no future leakage).  
   * Uses a triangular mask $M$ where:

     $$
     M_{ij} =
     \begin{cases}
     0, & j \le i,\\
     -\infty, & j > i
     \end{cases}
     $$

     so that the softmax ignores future tokens.


2. Encoder–Decoder Attention

   * Attends to encoder outputs (useful in translation or other paired tasks).
   * Queries come from the decoder, keys/values from the encoder.

3. Feedforward Network

   * Two linear transformations with ReLU or GELU:
     $$
     \text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
     $$

Each sub-layer includes:

- Residual connection
- Layer normalization

#### The Transformer Decoder Block

For each token representation $x_t$:

$$
\begin{aligned}
z_1 &= \text{LayerNorm}(x_t + \text{SelfAttention}(x_t)) \
z_2 &= \text{LayerNorm}(z_1 + \text{CrossAttention}(z_1, E)) \
y_t &= \text{LayerNorm}(z_2 + \text{FFN}(z_2))
\end{aligned}
$$

Where $E$ are encoder outputs (for tasks like translation).
In pure language models (like GPT), the encoder is omitted, only self-attention is used.

#### How It Works (Plain Language)

Imagine writing a story.
At each word, you look back at everything you've written so far, but not ahead, to decide what comes next.
You also consult a memory of "facts" (from an encoder or context).
The decoder does the same, combining self-attention and feedforward reasoning to build meaning word by word.

#### Step-by-Step Summary

| Step | Sub-Layer                 | Function                              |
| ---- | ------------------------- | ------------------------------------- |
| 1    | Masked Self-Attention     | Look at previous tokens               |
| 2    | Encoder–Decoder Attention | Look at encoded input (optional)      |
| 3    | Feedforward               | Nonlinear transformation              |
| 4    | Residual + Normalization  | Stability and gradient flow           |
| 5    | Output Projection         | Map hidden state to vocabulary logits |

#### Example: Auto-Regressive Generation

At inference time:

1. Start with `<BOS>` (beginning of sequence).
2. Predict next token using softmax over output logits:
   $$
   P(y_t \mid y_{<t}) = \text{softmax}(W_o h_t)
   $$
3. Append $y_t$ and feed back into model.
4. Repeat until `<EOS>` or length limit.

#### Tiny Code (Simplified)

Python (NumPy prototype)

```python
import numpy as np

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum(axis=-1, keepdims=True)

def attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    if mask is not None:
        scores += mask  # apply -inf for masked positions
    weights = softmax(scores)
    return weights @ V

def transformer_decoder_step(Q, K, V, mask):
    context = attention(Q, K, V, mask)
    return context + Q  # simple residual form
```

C (Outline)

```c
// For each token t: compute Q,K,V; mask future positions; apply dot-product attention.
// Add residual and normalization steps (omitted for brevity).
```

#### Why It Matters

- Replaces recurrence with attention → parallelizable and faster.
- Captures global context across long sequences.
- Scales beautifully → supports huge models (billions of parameters).
- Foundation of GPT, BERT, T5, LLaMA, etc.
- Language, vision, audio, and multimodal unification.

The decoder architecture is the reason large language models can write essays, code, and poetry by simply attending over massive context windows.

#### A Gentle Proof (Why It Works)

Self-attention builds pairwise dependencies between all positions in a sequence.
Unlike RNNs that rely on chain recurrence:

$$
h_t = f(h_{t-1}, x_t)
$$

the Transformer computes all $h_t$ simultaneously via attention weights.
The model thus learns contextual relationships in a single step, enabling efficient long-range reasoning.

The mask ensures causality, no "peeking" at future tokens.

#### Try It Yourself

1. Implement a 2-layer decoder-only Transformer on a toy dataset.
2. Compare with an RNN: note faster convergence and better long-term recall.
3. Visualize attention weights during decoding, see how tokens attend backward.
4. Experiment with different mask sizes (context windows).
5. Try greedy vs beam search decoding.

#### Test Cases

| Model   | Task               | Type            | Decoder Only? |
| ------- | ------------------ | --------------- | ------------- |
| GPT     | Text generation    | Language        | Yes           |
| T5      | Translation        | Encoder–Decoder | No            |
| Whisper | Speech recognition | Encoder–Decoder | No            |
| LLaMA   | Chat model         | Language        | Yes           |

#### Complexity

- Time: $O(T^2 d)$ per layer
- Space: $O(T^2)$
- Parallelism: fully parallel per token

The Transformer Decoder is the modern storyteller, reading everything it's written, remembering context precisely, and weaving coherent sequences without ever looping back in time.

### 949. Seq2Seq with Attention

Seq2Seq with Attention combines two powerful ideas, sequence-to-sequence modeling and the attention mechanism, to enable flexible mapping from one sequence to another, especially when input and output lengths differ.
It marked a major breakthrough in neural machine translation, later evolving into the foundation of modern Transformer architectures.

#### What Problem Are We Solving?

In classic sequence-to-sequence (Seq2Seq) models using RNNs:

- The encoder reads the entire input sequence into a fixed-length vector.
- The decoder generates outputs step by step, based only on that vector.

Problem: long sequences cause information bottlenecks, early tokens are forgotten, context blurs.

The solution: add attention, so the decoder can look back at *all* encoder states, not just the final one.

#### The Core Idea

Seq2Seq with Attention has two main components:

1. Encoder
   Reads the input sequence $(x_1, \dots, x_T)$ and produces a sequence of hidden states:
   $$
   h_i = \text{Encoder}(x_i, h_{i-1})
   $$

2. Decoder
   Generates each output token $y_t$ while attending to all encoder states via attention weights $\alpha_{t,i}$.

At each decoding step $t$:

- Compute alignment scores:
  $$
  e_{t,i} = \text{score}(s_{t-1}, h_i)
  $$
- Normalize with softmax:
  $$
  \alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_j \exp(e_{t,j})}
  $$
- Compute context vector:
  $$
  c_t = \sum_i \alpha_{t,i} h_i
  $$
- Update the decoder state:
  $$
  s_t = f(s_{t-1}, y_{t-1}, c_t)
  $$
- Predict next token:
  $$
  P(y_t \mid y_{<t}, X) = \text{softmax}(W[s_t; c_t])
  $$

#### How It Works (Plain Language)

Think of translation:
The encoder reads "Le chat dort." → it builds memory for each word.
The decoder starts generating "The cat sleeps." and, at each step, looks back at the encoder's memory, focusing attention where it's most relevant.

When producing "cat," attention peaks on "chat."
When producing "sleeps," attention peaks on "dort."

This selective focusing eliminates the bottleneck and gives the model interpretability.

#### Step-by-Step Summary

| Step | Component | Description                                                          |
| ---- | --------- | -------------------------------------------------------------------- |
| 1    | Encoder   | Reads input sequence, produces hidden states                         |
| 2    | Attention | Computes relevance of each encoder state to decoder state            |
| 3    | Context   | Weighted sum of encoder hidden states                                |
| 4    | Decoder   | Uses previous token, context, and hidden state to predict next token |
| 5    | Output    | Softmax over vocabulary for next word                                |

#### Example: Simple Translation

Input: "je t'aime"
Encoder outputs hidden states $h_1, h_2, h_3$.
Decoder generates:

| Step | Output | Attention Peaks |
| ---- | ------ | --------------- |
| 1    | I      | $h_1$ (je)      |
| 2    | love   | $h_3$ (aime)    |
| 3    | you    | $h_2$ (t')      |

This dynamic alignment enables fluent translations without explicit word alignment.

#### Tiny Code (Simplified)

Python

```python
import numpy as np

def softmax(x): return np.exp(x) / np.exp(x).sum()

def attention(decoder_state, encoder_states):
    scores = encoder_states @ decoder_state
    weights = softmax(scores)
    context = (encoder_states.T @ weights).T
    return context, weights

def seq2seq_step(prev_state, prev_output, encoder_states):
    context, _ = attention(prev_state, encoder_states)
    new_state = np.tanh(prev_state + context + prev_output)
    return new_state
```

C (Outline)

```c
// For each decoder step:
// 1. Compute scores = dot(decoder_state, encoder_states)
// 2. Apply softmax to get attention weights
// 3. Compute weighted context vector
// 4. Combine with decoder RNN for next output
```

#### Why It Matters

- Solves information bottleneck in basic Seq2Seq.
- Improves translation, summarization, and speech tasks.
- Interpretable via attention heatmaps.
- Precursor to Transformers: replaces recurrence with self-attention.
- Flexible for variable-length input/output.

The attention mechanism transformed sequence learning from "compress and decode" to "consult and compose."

#### A Gentle Proof (Why It Works)

Without attention, the conditional probability is:

$$
P(y_t \mid y_{<t}, X) = f(s_{t-1}, h_T)
$$

With attention, we instead marginalize over all encoder states:

$$
P(y_t \mid y_{<t}, X) = f\left(s_{t-1}, \sum_i \alpha_{t,i} h_i\right)
$$

This turns a single-context model into a context distribution model, more expressive and differentiable, improving gradient flow and alignment learning.

#### Try It Yourself

1. Train a small Seq2Seq model on short English–French pairs.
2. Visualize attention weights per decoding step.
3. Increase sentence length, see how attention preserves context.
4. Compare with no-attention baseline.
5. Experiment with Bahdanau (additive) vs Luong (dot-product) scoring.

#### Test Cases

| Task               | Input              | Output         | Attention Role           |
| ------------------ | ------------------ | -------------- | ------------------------ |
| Translation        | "guten morgen"     | "good morning" | Aligns words             |
| Summarization      | Article            | Summary        | Focuses on main phrases  |
| Speech recognition | Audio frames       | Text           | Time alignment           |
| Question answering | Context + question | Answer         | Focuses on relevant span |

#### Complexity

- Time: $O(T_x T_y d)$
- Space: $O(T_x d)$
- Improvement: parallelizable and interpretable

Seq2Seq with Attention was the first neural model to "look while it speaks", an elegant bridge between memory and focus that opened the path to the Transformer revolution.

### 950. Pointer Network

The Pointer Network is a fascinating twist on the standard sequence-to-sequence architecture, instead of predicting tokens from a fixed vocabulary, it points to elements in the input sequence as outputs.

It's especially useful when the output is a reordering or subset of inputs, such as in sorting, routing, or combinatorial optimization problems.

#### What Problem Are We Solving?

Ordinary Seq2Seq models generate outputs from a fixed-size vocabulary.
But what if the output must refer to specific positions in the input?
For example:

- Sorting numbers (output is a permutation of input indices)
- Solving Traveling Salesman Problem (output is sequence of city indices)
- Extractive summarization (output is a subset of tokens)

A normal softmax over a fixed vocabulary cannot express this, 
we need a model that can dynamically select input elements as outputs.

#### The Core Idea

Pointer Networks reuse the attention mechanism as a pointer instead of a weighting function.
At each decoding step, the model computes attention over the encoder inputs and selects one position as the next output.

For input sequence $X = (x_1, x_2, \dots, x_n)$ and decoder state $s_t$:

1. Score each input:
   $$
   e_{t,i} = v_a^\top \tanh(W_1 h_i + W_2 s_t)
   $$

2. Normalize with softmax:
   $$
   p_{t,i} = \frac{\exp(e_{t,i})}{\sum_j \exp(e_{t,j})}
   $$

3. The output at step $t$ is the index with highest probability:
   $$
   y_t = \arg\max_i p_{t,i}
   $$

So instead of generating tokens, it "points" to the relevant input position.

#### How It Works (Plain Language)

Imagine you have a list of cities and you want the model to plan the shortest route.
Rather than generating city names, it points to the next city index in the input list, like a finger tracing an optimal tour.

Each step's attention distribution acts like a probability map over input positions.

#### Step-by-Step Summary

| Step | Component | Description                                       |
| ---- | --------- | ------------------------------------------------- |
| 1    | Encoder   | Reads all input elements into hidden states $h_i$ |
| 2    | Decoder   | Maintains a recurrent state $s_t$                 |
| 3    | Attention | Computes scores $e_{t,i}$ for each input position |
| 4    | Softmax   | Converts to probability distribution $p_{t,i}$    |
| 5    | Output    | Picks the most probable input index               |

#### Example: Sorting Three Numbers

Input: `[3, 1, 2]`
Encoder hidden states represent each number.
Decoder outputs indices `[2, 3, 1]`, corresponding to sorted order `[1, 2, 3]`.

| Step | Decoder Output | Selected Input | Meaning         |
| ---- | -------------- | -------------- | --------------- |
| 1    | 2              | 1              | smallest number |
| 2    | 3              | 2              | next smallest   |
| 3    | 1              | 3              | largest         |

#### Tiny Code (Simplified)

Python

```python
import numpy as np

def pointer_network_step(encoder_h, decoder_s):
    scores = np.tanh(encoder_h @ W1 + decoder_s @ W2)
    scores = scores @ v_a
    probs = np.exp(scores) / np.exp(scores).sum()
    pointer = np.argmax(probs)
    return pointer, probs
```

C (Outline)

```c
// Compute attention scores between decoder state and encoder states
// Apply softmax over input positions to get probabilities
// Output the index with highest probability
```

#### Why It Matters

- Dynamic output size: no need for fixed vocabularies.
- Perfect for structured tasks: sorting, matching, routing.
- Interpretable: attention weights show selection probabilities.
- Bridges deep learning and combinatorial optimization.

The model learns to select rather than generate, 
a subtle shift that opens deep learning to problems in graph theory and operations research.

#### A Gentle Proof (Why It Works)

In normal attention-based Seq2Seq models, the attention weights $\alpha_{t,i}$ act as context weights:

$$
c_t = \sum_i \alpha_{t,i} h_i
$$

In a Pointer Network, these same weights form the output distribution:

$$
P(y_t = i \mid X, y_{<t}) = \alpha_{t,i}
$$

Since $\alpha_{t,i}$ depends on both input representations and decoder state, the model learns differentiable selection behavior through standard backpropagation.

#### Try It Yourself

1. Train a Pointer Network to sort small lists of integers.
2. Visualize the attention weights as "pointers" over time.
3. Extend to Traveling Salesman Problem (TSP), the model outputs city order.
4. Try extractive summarization, selecting key sentences from text.
5. Compare with a fixed-vocabulary Seq2Seq baseline.

#### Test Cases

| Task               | Input              | Output              | Model Behavior             |
| ------------------ | ------------------ | ------------------- | -------------------------- |
| Sorting            | [5, 1, 4]          | [2, 3, 1]           | Picks smallest-first order |
| TSP                | 5 city coordinates | Sequence of indices | Routes cities efficiently  |
| Text summarization | Sentences          | Selected subset     | Extracts key lines         |
| Question answering | Context words      | Word index          | Points to answer span      |

#### Complexity

- Time: $O(n^2)$ (attention over all inputs)
- Space: $O(n^2)$
- Output flexibility: up to length of input

Pointer Networks show that deep learning models don't always need to invent new words, sometimes, they just need to *point* to what's already there.

# Section 96. Metaheuristics 

### 951. Genetic Algorithm (GA)

The Genetic Algorithm (GA) is an evolutionary search technique inspired by natural selection.
It explores a large solution space by evolving a population of candidates through selection, crossover, and mutation, just like biological evolution improves species over generations.

#### What Problem Are We Solving?

Many real-world problems, scheduling, design, routing, optimization, have search spaces that are too large or complex for exact algorithms.
Genetic Algorithms offer a stochastic, adaptive method to find *good enough* solutions without brute force.

They're particularly useful when:

- The problem has no gradient or nonlinear objective
- The solution space is discrete or combinatorial
- We only need a near-optimal answer quickly

#### The Core Idea

A GA maintains a population of candidate solutions (chromosomes).
Each solution is represented as a string (e.g., bits, numbers, symbols).
Over generations, the population "evolves" toward better fitness.

Main steps:

1. Initialization, create random population $P_0$
2. Selection, choose better individuals for reproduction
3. Crossover, combine pairs to create offspring
4. Mutation, randomly tweak offspring to preserve diversity
5. Replacement, form the next generation $P_{t+1}$
6. Repeat until convergence or max generations

#### The Genetic Cycle (Plain Language)

1. Start randomly, guess a bunch of solutions.
2. Score them, use a fitness function $f(x)$ to measure quality.
3. Breed survivors, the fittest "parents" mate to create children.
4. Mix and mutate, small random changes introduce variation.
5. Keep the best, select top candidates for the next generation.
6. Repeat, evolution continues until performance stabilizes.

#### Mathematical View

Let:

- $x_i^{(t)}$ be the $i$-th individual at generation $t$
- $f(x_i^{(t)})$ be its fitness score

The population evolves according to:

$$
P^{(t+1)} = \text{Select}\big(\text{Mutate}(\text{Crossover}(P^{(t)}))\big)
$$

The process balances exploration (via mutation) and exploitation (via selection).

#### Step-by-Step Example (Binary Optimization)

Goal: maximize $f(x) = x^2$, where $x$ is a 5-bit binary number.

| Generation | Population                   | Fitness             | Selected       | Offspring      | Mutation       |
| ---------- | ---------------------------- | ------------------- | -------------- | -------------- | -------------- |
| 0          | [01011, 10000, 00101, 11100] | [121, 256, 25, 784] | [11100, 10000] | [11100, 10000] | [11110, 10100] |
| 1          | [11110, 10100, …]            | [900, 400, …]       | …              | …              | …              |

After several generations, the population converges toward $x=11111$ (31), the global optimum.

#### Tiny Code

Python

```python
import random

def fitness(x):
    return x * x

def mutate(x):
    mask = 1 << random.randint(0, 4)
    return x ^ mask

def crossover(a, b):
    point = random.randint(1, 4)
    mask = (1 << point) - 1
    return (a & mask) | (b & ~mask)

# Initialization
pop = [random.randint(0, 31) for _ in range(4)]
for gen in range(10):
    pop = sorted(pop, key=fitness, reverse=True)
    a, b = pop[:2]
    child = mutate(crossover(a, b))
    pop[-1] = child
    print(f"Gen {gen}: best {a}, fitness={fitness(a)}")
```

C (Sketch)

```c
// Represent each candidate as an unsigned int.
// Apply crossover by bitmask, mutation by flipping random bits.
// Evaluate fitness and keep top individuals each generation.
```

#### Why It Matters

- Works for non-differentiable, discrete, or black-box problems.
- Naturally parallel and stochastic.
- Simple to implement and adapt.
- Used in design optimization, route planning, neural architecture search, and more.

#### A Gentle Proof (Why It Works)

Genetic Algorithms approximate stochastic hill climbing with population diversity.
Under the Schema Theorem, patterns (schemata) representing good partial solutions are exponentially propagated if they contribute above-average fitness:

$$
E[m(H, t+1)] \ge m(H, t) \frac{f(H)}{\bar{f}} (1 - p_c - p_m)
$$

where:

- $m(H, t)$ is number of schema instances
- $f(H)$ is average schema fitness
- $\bar{f}$ is population mean fitness
- $p_c, p_m$ are crossover and mutation probabilities

Over time, beneficial schemata dominate the population.

#### Try It Yourself

1. Implement GA for function maximization $f(x) = \sin(x)$ on $[0, 2\pi]$.
2. Vary mutation rate and observe diversity.
3. Compare roulette-wheel vs tournament selection.
4. Visualize convergence of best fitness over generations.
5. Use GA to evolve weights for a tiny neural network.

#### Test Cases

| Task                    | Representation    | Fitness Function    | Notes             |
| ----------------------- | ----------------- | ------------------- | ----------------- |
| Bit-string optimization | Binary            | Problem-defined     | Classic           |
| TSP                     | City sequence     | Inverse path length | Popular benchmark |
| Symbolic regression     | Expression tree   | Error vs data       | GP variant        |
| Scheduling              | Job order         | Minimize delay      | Discrete          |
| Neural architecture     | Network structure | Validation accuracy | Meta-learning     |

#### Complexity

- Time: $O(P \cdot G \cdot C)$

  * $P$: population size
  * $G$: generations
  * $C$: fitness cost
- Space: $O(P)$

Genetic Algorithms remind us that progress doesn't need precision, 
it can emerge from variation, competition, and persistence.

### 952. Simulated Annealing (SA)

Simulated Annealing is a probabilistic optimization algorithm inspired by the physical process of heating and slowly cooling metal to remove defects.
It's a single-solution search method that balances exploration (trying new solutions) and exploitation (keeping good ones) using controlled randomness.

#### What Problem Are We Solving?

Many optimization problems have many local minima, places where the solution looks good but isn't globally optimal.
A simple hill-climbing algorithm can get stuck there.

Simulated Annealing solves this by occasionally accepting worse solutions early on, to escape local traps, and then gradually reducing randomness to settle into a good region.

It's especially useful when:

- The search space is large or nonconvex
- Gradients are unavailable or noisy
- Deterministic methods fail due to local minima

#### The Core Idea

The algorithm mimics how materials anneal, heat, then cool slowly so particles settle into a low-energy (stable) state.

At each iteration:

1. Propose a new solution $x'$ from the current one $x$.
2. Compute the energy (objective) difference $\Delta E = f(x') - f(x)$.
3. Accept the new solution if:

   * $\Delta E < 0$ (better), or
   * With probability $p = \exp(-\Delta E / T)$ (worse but allowed)
4. Gradually lower the temperature $T$ → reduces randomness.
5. Stop when $T$ is very small or the solution stabilizes.

#### How It Works (Plain Language)

Imagine a ball rolling on a hilly landscape.
At high "temperature," it bounces around freely, sometimes climbing hills.
As it cools, its movement becomes more restricted, until it finally rests in a deep valley, ideally the global minimum.

Early randomness helps exploration, later cooling helps convergence.

#### Mathematical Formulation

Let $f(x)$ be the energy (objective function) to minimize.
At iteration $t$:

1. Propose new candidate $x'$ by perturbing $x$.
2. Compute $\Delta E = f(x') - f(x)$.
3. Accept $x'$ with probability:

$$
P(\text{accept}) =
\begin{cases}
1, & \text{if } \Delta E < 0,\\
\exp(-\Delta E / T_t), & \text{otherwise.}
\end{cases}
$$


4. Update temperature $T_t = \alpha T_{t-1}$, where $0 < \alpha < 1$ is cooling rate.

#### Step-by-Step Example

Goal: minimize $f(x) = x^2 + 10 \sin(x)$

| Iteration | Current $x$ | Candidate $x'$ | $\Delta E$ | $T$  | Accept?                         |
| --------- | ----------- | -------------- | ---------- | ---- | ------------------------------- |
| 1         | 2.0         | 3.1            | +4.7       | 1.0  | Yes (random accept)             |
| 2         | 3.1         | 1.5            | -6.0       | 0.9  | Yes                             |
| 3         | 1.5         | 1.9            | +0.5       | 0.81 | Maybe (depends on $e^{-0.5/T}$) |

Over time, acceptance of worse moves becomes rarer.

#### Tiny Code

Python

```python
import math, random

def f(x): return x2 + 10 * math.sin(x)

x = random.uniform(-10, 10)
T = 1.0
alpha = 0.95

for i in range(1000):
    x_new = x + random.uniform(-1, 1)
    dE = f(x_new) - f(x)
    if dE < 0 or random.random() < math.exp(-dE / T):
        x = x_new
    T *= alpha

print("Best x:", x, "f(x):", f(x))
```

C (Sketch)

```c
// Initialize x randomly, set T = 1
// Loop: propose x', compute dE = f(x') - f(x)
// if dE < 0 or exp(-dE/T) > rand(), accept x'
// gradually reduce T
```

#### Why It Matters

- Escapes local minima, a key advantage over greedy methods.
- Works even on non-smooth, discontinuous, or discrete problems.
- Simple to implement with minimal tuning.
- Useful in scheduling, circuit design, path planning, and layout optimization.

#### A Gentle Proof (Why It Works)

Simulated Annealing is rooted in Markov Chain Monte Carlo (MCMC) theory.
If the temperature decreases slowly enough, the algorithm converges (in probability) to the global minimum of $f(x)$.

At equilibrium, the system samples states according to the Boltzmann distribution:

$$
P(x) \propto \exp(-f(x) / T)
$$

As $T \to 0$, probability mass concentrates at the global optimum.

#### Try It Yourself

1. Minimize $f(x) = x^2 + 10\sin(x)$ and visualize the trajectory.
2. Experiment with different cooling schedules:

   * Exponential: $T_t = \alpha^t T_0$
   * Linear: $T_t = T_0 / (1 + \beta t)$
   * Logarithmic: $T_t = T_0 / \log(1 + t)$
3. Plot acceptance rate vs temperature.
4. Apply SA to the Traveling Salesman Problem.

#### Test Cases

| Problem               | Search Space | Notes                      |
| --------------------- | ------------ | -------------------------- |
| Function minimization | Continuous   | Simple benchmark           |
| TSP                   | Permutations | Classic combinatorial test |
| Job scheduling        | Discrete     | Large local minima count   |
| Neural net weights    | Continuous   | Experimental metaheuristic |

#### Complexity

- Time: $O(N \times k)$

  * $N$: iterations
  * $k$: cost of evaluating $f(x)$
- Space: $O(1)$
- Convergence depends on cooling schedule, slower = better.

Simulated Annealing is like controlled chaos, it wanders aimlessly at first, then slowly finds order, cooling into a near-perfect solution one probability drop at a time.

### 953. Tabu Search

Tabu Search is a metaheuristic that guides a local search algorithm to escape local minima by remembering, and avoiding, recently visited solutions.
It adds short-term memory to the optimization process, helping the algorithm explore new regions of the search space without cycling back.

#### What Problem Are We Solving?

Local search algorithms (like hill climbing) can get trapped in local optima, endlessly revisiting the same or similar solutions.
Tabu Search introduces a memory-based strategy to overcome this, it forbids (makes *tabu*) moves that undo recent progress.

This is useful in hard combinatorial problems such as:

- Traveling Salesman Problem (TSP)
- Scheduling and timetabling
- Graph coloring
- Resource allocation

#### The Core Idea

Tabu Search enhances simple hill climbing with three key mechanisms:

1. Tabu List (Memory)
   A short-term memory storing recent moves or attributes to prevent cycling.

2. Aspiration Criteria
   Allows overriding a tabu restriction if the move yields a significantly better solution.

3. Neighborhood Exploration
   Systematically evaluates nearby solutions (the neighborhood) and picks the best non-tabu one.

The algorithm continues until a stopping condition is met, like reaching a time limit or no improvement after several iterations.

#### How It Works (Plain Language)

Imagine you're hiking through hills in the dark.
You can only feel the ground beneath your feet, so you take the steepest path uphill.
But if you always do that, you'll end up stuck on a small hill.

Tabu Search helps by keeping track of where you've already been, forbidding you from stepping back too soon, and occasionally allowing a risky detour if it looks promising.

#### Mathematical Formulation

Let:

- $S$ be the search space
- $f(s)$ the cost (to minimize)
- $N(s)$ the neighborhood of $s$

At iteration $t$:

1. From current solution $s_t$, generate neighborhood $N(s_t)$.
2. Exclude tabu moves, those stored in the tabu list $T_t$.
3. Select the best candidate:
   $$
   s_{t+1} = \arg\min_{s' \in N(s_t) \setminus T_t} f(s')
   $$
4. Update tabu list:
   $$
   T_{t+1} = (T_t \cup {\text{move}(s_t, s_{t+1})}) - \text{expired entries}
   $$
5. If $f(s_{t+1})$ improves global best, update it.

#### Step-by-Step Example (TSP)

Goal: find shortest route through cities A–E.

| Iteration | Current Route | Cost | Tabu Moves | Best Move | Next Route             |
| --------- | ------------- | ---- | ---------- | --------- | ---------------------- |
| 1         | A–B–C–D–E     | 34   | (swap A–B) | swap C–D  | A–B–D–C–E              |
| 2         | A–B–D–C–E     | 31   | (swap C–D) | swap B–D  | A–D–B–C–E              |
| 3         | A–D–B–C–E     | 29   | (swap B–D) | swap A–D  | A–B–C–D–E (tabu, skip) |
| 4         | …             | …    | …          | …         | …                      |

The algorithm avoids returning to previous routes too early, exploring new configurations.

#### Tiny Code

Python (Simplified)

```python
import random

def tabu_search(init, neighbor_fn, cost_fn, max_iter=100, tabu_len=5):
    current = init
    best = init
    tabu_list = []

    for _ in range(max_iter):
        neighbors = neighbor_fn(current)
        candidates = [n for n in neighbors if n not in tabu_list]

        if not candidates:
            candidates = neighbors  # if all tabu, ignore restriction temporarily

        next_candidate = min(candidates, key=cost_fn)
        if cost_fn(next_candidate) < cost_fn(best):
            best = next_candidate

        tabu_list.append(next_candidate)
        if len(tabu_list) > tabu_len:
            tabu_list.pop(0)

        current = next_candidate
    return best
```

C (Sketch)

```c
// Store recent moves in a fixed-size array (tabu list).
// Each iteration, generate all neighbors, skip tabu ones.
// Choose best allowed neighbor; update tabu list.
```

#### Why It Matters

- Avoids cycles and premature convergence.
- Can escape local minima without random jumps.
- Adapts to many combinatorial optimization problems.
- Often finds near-optimal solutions efficiently.

It's like giving hill climbing a memory, so it doesn't make the same mistakes twice.

#### A Gentle Proof (Why It Works)

Tabu Search maintains a diversity of explored regions by enforcing a short-term prohibition of recent states.
This forces the search trajectory to move outward in the solution graph.
The aspiration criterion ensures that even if a tabu move leads to a global improvement, it can be accepted, ensuring convergence toward strong candidates.

Over time, the algorithm approximates a balance between intensification (local search near good regions) and diversification (exploring new ones).

#### Try It Yourself

1. Implement a Tabu Search for the 8-queens problem.
2. Vary tabu list length, short lists cause cycling, long lists slow progress.
3. Add an aspiration rule that allows tabu moves if cost improves best-so-far.
4. Visualize path of cost per iteration.
5. Compare to hill climbing and simulated annealing.

#### Test Cases

| Task           | Representation | Tabu Memory          | Notes                   |
| -------------- | -------------- | -------------------- | ----------------------- |
| TSP            | Route order    | Recent swaps         | Classic benchmark       |
| Job scheduling | Job order      | Recent exchanges     | Industrial optimization |
| Graph coloring | Node colors    | Recent color changes | NP-hard                 |
| Sudoku         | Grid state     | Cell assignments     | Constraint satisfaction |

#### Complexity

- Time: $O(n \cdot |N(s)|)$ per iteration
- Space: $O(|T|)$ (tabu list length)
- Convergence: depends on neighborhood size and tabu length

Tabu Search is a clever balance between memory and curiosity, it remembers enough to avoid loops but forgets just enough to keep exploring.

### 954. Particle Swarm Optimization (PSO)

Particle Swarm Optimization (PSO) is a population-based metaheuristic inspired by how flocks of birds or schools of fish move together toward food sources.
Each "particle" represents a potential solution that moves through the search space, guided by its own experience and by the best performers in the swarm.

#### What Problem Are We Solving?

We often face optimization problems where:

- The objective function is nonlinear or non-differentiable
- The landscape has many local optima
- Gradient information is unavailable

PSO provides a derivative-free, parallel, and adaptive way to explore such spaces efficiently.
It's widely used in control, parameter tuning, neural network training, and design optimization.

#### The Core Idea

Each particle $i$ maintains:

- Position $x_i$, current solution
- Velocity $v_i$, direction and step size
- Personal best $p_i$, best position found so far
- Global best $g$, best among all particles

At every iteration:

1. Update velocity (momentum + attraction to bests)
2. Update position
3. Evaluate new position and update $p_i$ and $g$ if improved

#### Mathematical Formulation

For particle $i$ at iteration $t$:

1. Velocity update:
   $$
   v_i(t+1) = \omega v_i(t)
   + c_1 r_1 (p_i - x_i(t))
   + c_2 r_2 (g - x_i(t))
   $$

2. Position update:
   $$
   x_i(t+1) = x_i(t) + v_i(t+1)
   $$

where

- $\omega$ is the inertia weight (controls exploration)
- $c_1, c_2$ are acceleration coefficients (self and social influence)
- $r_1, r_2$ are random numbers in $[0,1]$

The balance between inertia and attraction determines how the swarm explores and converges.

#### How It Works (Plain Language)

Imagine a flock of birds looking for food on a landscape.
Each bird remembers the best spot it has found and also watches the best spot found by any bird.
They all adjust their flight, slightly toward where they found food before, and slightly toward where the best bird is going.
Over time, the swarm naturally gathers around the optimal area.

#### Step-by-Step Example

Goal: minimize $f(x) = x^2 + 3\sin(x)$

| Particle | Initial $x$ | Best $p_i$ | Global $g$ | Update Rule |
| -------- | ----------- | ---------- | ---------- | ----------- |
| 1        | 2.5         | 2.5        | 1.7        | Moves left  |
| 2        | -1.0        | -1.0       | 1.7        | Moves right |
| 3        | 3.1         | 3.1        | 1.7        | Moves left  |
| …        | …           | …          | …          | …           |

Eventually, all particles converge near $x = 1.7$ (the global minimum).

#### Tiny Code

Python

```python
import random
import math

def f(x): return x2 + 3 * math.sin(x)

num_particles = 10
x = [random.uniform(-5, 5) for _ in range(num_particles)]
v = [0 for _ in range(num_particles)]
p = x[:]
g = min(p, key=f)

w, c1, c2 = 0.7, 1.5, 1.5

for t in range(100):
    for i in range(num_particles):
        r1, r2 = random.random(), random.random()
        v[i] = w*v[i] + c1*r1*(p[i]-x[i]) + c2*r2*(g-x[i])
        x[i] += v[i]
        if f(x[i]) < f(p[i]):
            p[i] = x[i]
        if f(x[i]) < f(g):
            g = x[i]

print("Best solution:", g, "f(x):", f(g))
```

C (Sketch)

```c
// Initialize arrays x[], v[], p[] for N particles.
// Loop: update v[i], x[i], evaluate f(x[i]),
// track global best g and local best p[i].
```

#### Why It Matters

- Requires no gradient, ideal for black-box problems.
- Fast, parallelizable, and robust.
- Naturally balances exploration (through inertia) and exploitation (through social learning).
- Effective across continuous and combinatorial spaces.

PSO has become a go-to algorithm for parameter optimization in engineering, AI, and scientific modeling.

#### A Gentle Proof (Why It Works)

The swarm's behavior emerges from feedback loops between velocity, position, and best memories.
When inertia is high ($\omega$ large), particles explore broadly.
When $\omega$ is small, attraction to $p_i$ and $g$ dominates, leading to convergence.

Stability analysis shows that under suitable $\omega, c_1, c_2$ values (commonly $\omega \in [0.4,0.9]$, $c_1=c_2=2$), the system converges probabilistically toward an equilibrium near the optimum.

#### Try It Yourself

1. Plot particle trajectories for $f(x)=x^2$.
2. Tune $\omega$:

   * Too high → oscillations
   * Too low → premature convergence
3. Compare PSO with Simulated Annealing on the same function.
4. Extend to 2D functions like Rastrigin or Rosenbrock.
5. Observe swarm dynamics visually.

#### Test Cases

| Task                  | Dimension | Objective        | Behavior           |
| --------------------- | --------- | ---------------- | ------------------ |
| Sphere function       | 1–10D     | $x^2$            | Fast convergence   |
| Rastrigin             | 2D        | Multi-modal      | Good global search |
| Neural network tuning | High-D    | Validation error | Smooth exploration |
| PID controller tuning | 3D        | Control error    | Stable convergence |

#### Complexity

- Time: $O(N_p \cdot T)$

  * $N_p$: number of particles
  * $T$: iterations
- Space: $O(N_p)$
- Convergence: depends on parameter balance and noise level

Particle Swarm Optimization is evolution without genes, a choreography of searchers learning from one another until they flock around the truth.

### 955. Ant Colony Optimization (ACO)

Ant Colony Optimization (ACO) is a metaheuristic inspired by how real ants find the shortest path to food sources using pheromone trails.
It turns collective behavior into a computational method for solving difficult combinatorial problems such as routing, scheduling, and network optimization.

#### What Problem Are We Solving?

Many combinatorial optimization problems, like the Traveling Salesman Problem (TSP) or network routing, are too large for exhaustive search.
We need algorithms that can efficiently explore and share information about promising paths.

Ant Colony Optimization provides a distributed probabilistic search that gradually strengthens good solutions through reinforcement.

#### The Core Idea

Ants explore possible solutions (paths) and communicate indirectly by depositing pheromones, which represent how desirable each choice is.
Over time:

- Shorter or better paths accumulate more pheromone
- Pheromones evaporate, reducing attraction to bad paths
- The colony converges on high-quality solutions

#### Mathematical Formulation

Each ant builds a path step by step based on probabilities determined by:

1. Pheromone trail strength $\tau_{ij}$ on edge $(i, j)$
2. Heuristic desirability $\eta_{ij}$ (e.g., inverse of distance)

The probability that an ant moves from $i$ to $j$ is:

$$
P_{ij} =
\frac{\tau_{ij}^\alpha \eta_{ij}^\beta}
{\sum_{k \in \text{allowed}} \tau_{ik}^\alpha \eta_{ik}^\beta}
$$

where:

- $\alpha$ controls the influence of pheromones
- $\beta$ controls the influence of heuristic information

After all ants construct solutions, pheromones are updated:

$$
\tau_{ij} \leftarrow (1 - \rho) \tau_{ij} + \sum_k \Delta \tau_{ij}^k
$$

where $\rho$ is evaporation rate, and
$$
\Delta \tau_{ij}^k =
\begin{cases}
\frac{Q}{L_k}, & \text{if ant } k \text{ used edge } (i,j),\\
0, & \text{otherwise.}
\end{cases}
$$


with $L_k$ being the total length (cost) of ant $k$'s path.

#### How It Works (Plain Language)

Picture a group of ants searching for food.
Initially, they wander randomly, leaving pheromone trails.
Ants that happen to find a short path back to the nest reinforce it with more pheromone.
Later ants prefer stronger trails, making short paths more likely to be reused.
Evaporation ensures exploration doesn't stop too soon.

Over time, the colony collectively "discovers" the best routes.

#### Step-by-Step Example (Traveling Salesman Problem)

| Step | Phase          | Description                                         |
| ---- | -------------- | --------------------------------------------------- |
| 1    | Initialization | Set pheromones $\tau_{ij} = \tau_0$ on all edges    |
| 2    | Construction   | Each ant builds a complete tour probabilistically   |
| 3    | Evaluation     | Compute length $L_k$ of each tour                   |
| 4    | Update         | Deposit pheromone $\propto 1/L_k$; evaporate $\rho$ |
| 5    | Repeat         | Until convergence or max iterations                 |

Ants indirectly collaborate through pheromone feedback, no direct communication needed.

#### Tiny Code

Python (Simplified TSP)

```python
import math, random

def distance(a, b): return math.hypot(a[0]-b[0], a[1]-b[1])

cities = [(0,0), (1,5), (5,2), (6,6)]
n = len(cities)
pheromone = [[1 for _ in range(n)] for _ in range(n)]
alpha, beta, rho, Q = 1.0, 3.0, 0.5, 100

def tour_length(tour):
    return sum(distance(cities[tour[i]], cities[tour[(i+1)%n]]) for i in range(n))

def choose_next(current, visited):
    probs = []
    for j in range(n):
        if j not in visited:
            tau = pheromone[current][j]alpha
            eta = (1 / (distance(cities[current], cities[j]) + 1e-9))beta
            probs.append((j, tau * eta))
    total = sum(p for _, p in probs)
    r = random.random() * total
    s = 0
    for j, p in probs:
        s += p
        if s >= r:
            return j

for iteration in range(100):
    tours = []
    for _ in range(5):  # ants
        tour = [random.randint(0, n-1)]
        while len(tour) < n:
            next_city = choose_next(tour[-1], tour)
            tour.append(next_city)
        tours.append(tour)

    # evaporate
    for i in range(n):
        for j in range(n):
            pheromone[i][j] *= (1 - rho)

    # deposit
    for tour in tours:
        L = tour_length(tour)
        for i in range(n):
            a, b = tour[i], tour[(i+1)%n]
            pheromone[a][b] += Q / L
```

#### Why It Matters

- Powerful for NP-hard problems (TSP, VRP, scheduling).
- Adaptable to dynamic environments (e.g., changing networks).
- Distributed and parallel by nature.
- Combines exploitation (pheromones) and exploration (random choice).

ACO is one of the most successful swarm intelligence algorithms in optimization and logistics.

#### A Gentle Proof (Why It Works)

Ants collectively approximate a reinforcement learning process.
Each pheromone update acts like a weighted reward signal:

$$
\tau_{ij}(t+1) = (1-\rho)\tau_{ij}(t) + \mathbb{E}\left[\frac{Q}{L}\right]
$$

Evaporation ensures forgetting of poor paths;
reinforcement strengthens good ones.
The colony asymptotically concentrates pheromones on optimal (or near-optimal) paths.

#### Try It Yourself

1. Implement ACO for a 10-city TSP.
2. Plot pheromone intensities after each iteration.
3. Tune $\alpha$, $\beta$, and $\rho$:

   * High $\alpha$ → exploitation dominates.
   * High $\beta$ → greedy heuristic bias.
   * Low $\rho$ → pheromone saturation.
4. Compare convergence speed for different parameters.
5. Apply to network routing or job sequencing.

#### Test Cases

| Problem         | Search Space     | Objective         | Notes                   |
| --------------- | ---------------- | ----------------- | ----------------------- |
| TSP             | Cities and edges | Shortest tour     | Canonical benchmark     |
| Job scheduling  | Tasks            | Minimize makespan | Industrial planning     |
| Graph coloring  | Nodes            | Fewer colors      | Constraint satisfaction |
| Network routing | Paths            | Minimize latency  | Dynamic version         |

#### Complexity

- Time: $O(m \cdot n^2)$ per iteration

  * $m$: number of ants
  * $n$: number of nodes
- Space: $O(n^2)$ (pheromone matrix)
- Convergence: strongly influenced by evaporation $\rho$

Ant Colony Optimization shows how *simple agents with local rules* can create intelligent, emergent global behavior, 
nature's optimization at its finest, reimagined in code.

### 956. Differential Evolution (DE)

Differential Evolution (DE) is a population-based optimization algorithm designed for continuous-valued functions.
It's simple, robust, and remarkably effective, a minimalist cousin of Genetic Algorithms that replaces complex crossover rules with a single elegant operation: *vector difference mutation.*

#### What Problem Are We Solving?

DE is used to solve continuous optimization problems where:

- The function is nonlinear or multimodal
- Derivatives are unavailable or unreliable
- We need a balance between exploration and precision

It's particularly effective for engineering design, neural tuning, and simulation calibration tasks where gradients are hard to compute.

#### The Core Idea

A DE population consists of candidate solutions (vectors) that evolve over generations by combining other individuals to create new trial vectors.

Each generation involves three key steps:

1. Mutation, create a trial vector by adding a scaled difference between two population vectors to a third.
2. Crossover, mix components of the trial with the current vector.
3. Selection, keep the better vector based on fitness.

This simple arithmetic process efficiently explores the space without needing derivatives.

#### Mathematical Formulation

Let:

- $x_i^{(t)}$ be the $i$-th individual at generation $t$
- $f(x)$ be the objective function to minimize

Then for each $x_i$:

1. Mutation:

$$
v_i = x_{r_1} + F \cdot (x_{r_2} - x_{r_3})
$$

where $r_1, r_2, r_3$ are distinct random indices and $F \in [0, 2]$ is a scaling factor controlling mutation strength.

2. Crossover:

For each dimension $j$:

$$
u_{i,j} =
\begin{cases}
v_{i,j}, & \text{if } r_j < C_r \text{ or } j = j_{\text{rand}},\\
x_{i,j}, & \text{otherwise.}
\end{cases}
$$

where $C_r$ is the crossover rate.

3. Selection:

$$
x_i^{(t+1)} =
\begin{cases}
u_i, & \text{if } f(u_i) < f(x_i^{(t)}),\\
x_i^{(t)}, & \text{otherwise.}
\end{cases}
$$


#### How It Works (Plain Language)

Imagine a group of explorers scattered across a mountain range (the search space).
Each explorer tries new directions by combining the positions of three others, effectively following the vector between them.
If a new position is better (lower altitude), it replaces the old one.
Over time, the group collectively drifts downhill toward the global minimum.

#### Step-by-Step Example

Goal: minimize $f(x, y) = x^2 + y^2$

| Step | Operation             | Description                            |
| ---- | --------------------- | -------------------------------------- |
| 1    | Initialize population | Random 2D points                       |
| 2    | Mutation              | $v_i = x_{r_1} + F(x_{r_2} - x_{r_3})$ |
| 3    | Crossover             | Mix $v_i$ with $x_i$                   |
| 4    | Selection             | Keep the lower $f(x)$                  |
| 5    | Repeat                | Until convergence                      |

After a few generations, all individuals converge near $(0, 0)$, the optimal point.

#### Tiny Code

Python

```python
import random

def f(x, y): return x2 + y2

F, CR, NP, D = 0.8, 0.9, 10, 2
pop = [[random.uniform(-5, 5) for _ in range(D)] for _ in range(NP)]

for gen in range(100):
    for i in range(NP):
        r1, r2, r3 = random.sample(range(NP), 3)
        v = [pop[r1][j] + F * (pop[r2][j] - pop[r3][j]) for j in range(D)]
        u = [v[j] if random.random() < CR else pop[i][j] for j in range(D)]
        if f(*u) < f(*pop[i]):
            pop[i] = u

best = min(pop, key=lambda p: f(*p))
print("Best:", best, "f(x):", f(*best))
```

C (Sketch)

```c
// Initialize population of D-dimensional vectors.
// For each individual:
//   Select r1, r2, r3 distinct.
//   Compute mutant vector v = x[r1] + F*(x[r2]-x[r3]).
//   Perform crossover, evaluate trial u.
//   Replace x[i] if f(u) < f(x[i]).
```

#### Why It Matters

- Few parameters, only $F$ and $C_r$.
- No derivatives required, works with black-box objectives.
- Handles noisy and discontinuous functions.
- Excellent global convergence on many continuous benchmarks.
- Simple yet powerful, often outperforms more complex methods.

Differential Evolution embodies the principle of "less is more", minimal rules, maximal performance.

#### A Gentle Proof (Why It Works)

The mutation rule ensures *directed diversity*:
differences between random individuals guide exploration toward new, potentially better regions.
This maintains population diversity and prevents premature convergence.

Under standard assumptions, DE converges stochastically to a global minimum as long as the mutation and selection preserve enough variance in the population.

#### Try It Yourself

1. Optimize $f(x, y) = (x - 3)^2 + (y + 2)^2$.
2. Tune $F$ and $C_r$, try $F=0.5, 0.9, 1.2$.
3. Track convergence speed with different population sizes.
4. Compare to Particle Swarm Optimization on the same function.
5. Apply DE to train a neural network's weights directly.

#### Test Cases

| Problem            | Domain      | Notes               |
| ------------------ | ----------- | ------------------- |
| Sphere             | Continuous  | Smooth convex       |
| Rastrigin          | Multimodal  | Tests global search |
| Ackley             | Nonlinear   | Harsh gradients     |
| Engineering design | Real-valued | Practical use case  |

#### Complexity

- Time: $O(NP \cdot D \cdot G)$

  * $NP$: population size
  * $D$: dimension
  * $G$: generations
- Space: $O(NP \cdot D)$
- Convergence: typically fast and stable for small parameter sets

Differential Evolution is optimization stripped to its essentials, 
a quiet but relentless drift through space, driven only by difference and selection.

### 957. Harmony Search

Harmony Search (HS) is a metaheuristic optimization algorithm inspired by the improvisation process of musicians seeking harmony.
Just as musicians adjust pitches to achieve a pleasing sound, the algorithm adjusts solution variables to minimize (or maximize) an objective function.

It's simple, flexible, and effective for both continuous and discrete optimization, especially when the search space is irregular or noisy.

#### What Problem Are We Solving?

Many optimization problems are like musical compositions: they involve balancing multiple variables (or notes) to achieve an optimal outcome.
Traditional algorithms can get stuck or require gradient information, but Harmony Search explores the space with creativity and variation, without derivatives.

It is widely used in:

- Engineering design optimization
- Scheduling and allocation problems
- Neural network parameter tuning
- Structural design

#### The Core Idea

Harmony Search is based on a harmony memory (HM), a collection of solution vectors (harmonies).
New solutions are generated by improvising from the memory, using three rules:

1. Memory consideration, pick a value from existing harmonies.
2. Pitch adjustment, fine-tune that value slightly.
3. Random selection, introduce entirely new values for diversity.

The best harmonies are kept, replacing the worst, iteratively improving the composition.

#### Mathematical Formulation

Let the optimization problem be:

$$
\min f(x_1, x_2, \dots, x_N)
$$

where each variable $x_i$ has range $[L_i, U_i]$.

At each iteration:

1. Initialize harmony memory (HM):
   $$
   HM = {x^{(1)}, x^{(2)}, \dots, x^{(HMS)}}
   $$
   where HMS = harmony memory size.

2. Improvise a new harmony $x' = (x'_1, x'_2, \dots, x'_N)$:
   For each variable $x_i$:

   * With probability HMCR (Harmony Memory Considering Rate), choose $x_i$ from $HM$.
   * Else, choose randomly from $[L_i, U_i]$.
   * With probability PAR (Pitch Adjusting Rate), modify it slightly:
     $$
     x'_i \leftarrow x'_i + \delta, \quad \delta \in [-bw, bw]
     $$
     where $bw$ is bandwidth (tuning parameter).

3. Evaluate $f(x')$.

4. Update HM, if $x'$ is better than the worst harmony, replace it.

Repeat until stopping criterion (iterations or convergence) is met.

#### How It Works (Plain Language)

Imagine a jazz band improvising.
Each musician (variable) listens to others and chooses a note (value) from past harmonies or invents a new one.
Occasionally, they tweak their pitch (fine-tune a parameter).
As they play, the overall sound (objective function) improves, converging toward harmony, the optimal solution.

#### Step-by-Step Example

Goal: minimize $f(x, y) = x^2 + y^2$

| Step | Action        | Description                   |
| ---- | ------------- | ----------------------------- |
| 1    | Initialize HM | 3 random pairs $(x, y)$       |
| 2    | Improvise     | Pick $x, y$ from HM or random |
| 3    | Adjust pitch  | Slightly modify $x, y$        |
| 4    | Evaluate      | Compute $f(x, y)$             |
| 5    | Update HM     | Keep top 3 harmonies          |

After several iterations, HM converges near $(0, 0)$, the global optimum.

#### Tiny Code

Python

```python
import random

def f(x, y): return x2 + y2

HMS, HMCR, PAR, bw = 3, 0.9, 0.3, 0.1
HM = [[random.uniform(-5, 5), random.uniform(-5, 5)] for _ in range(HMS)]

for _ in range(1000):
    new = []
    for i in range(2):
        if random.random() < HMCR:
            new.append(random.choice(HM)[i])
            if random.random() < PAR:
                new[i] += random.uniform(-bw, bw)
        else:
            new.append(random.uniform(-5, 5))
    if f(*new) < max(HM, key=lambda h: f(*h), default=new):
        HM[-1] = new
    HM.sort(key=lambda h: f(*h))

best = HM[0]
print("Best:", best, "f(x):", f(*best))
```

C (Sketch)

```c
// Store harmony memory as 2D array.
// Generate new harmony using HMCR, PAR, and bw parameters.
// Replace worst harmony if new one is better.
```

#### Why It Matters

- Derivative-free optimization.
- Balance between memory and creativity.
- Few parameters: HMCR, PAR, and bw.
- Effective for complex, nonlinear, multi-modal problems.
- Naturally supports multi-objective extensions.

Harmony Search captures the essence of *explore, refine, and remember*, just like a real musician perfecting a melody.

#### A Gentle Proof (Why It Works)

Harmony Search works through a stochastic combination of intensification (using memory) and diversification (random variation).
The convergence of HM toward optimal solutions follows from probabilistic sampling, over time, high-fitness harmonies dominate the population.

With a high HMCR and moderate PAR, the algorithm maintains balance between reuse and innovation, preventing stagnation while refining solutions.

#### Try It Yourself

1. Optimize $f(x, y) = x^2 + y^2 + 3\sin(2x)$ using HS.
2. Vary HMCR (0.5–0.95) and PAR (0.1–0.5), observe exploration vs exploitation.
3. Use larger HM for higher-dimensional problems.
4. Compare with Genetic Algorithm and Differential Evolution.
5. Apply to parameter tuning (e.g., ML hyperparameters).

#### Test Cases

| Problem       | Domain     | Notes                          |
| ------------- | ---------- | ------------------------------ |
| Sphere        | Continuous | Simple benchmark               |
| Rastrigin     | Nonlinear  | Multi-modal surface            |
| Scheduling    | Discrete   | Combinatorial use              |
| Neural tuning | Continuous | Efficient with few evaluations |

#### Complexity

- Time: $O(HMS \times N \times T)$

  * $HMS$: harmony memory size
  * $N$: number of variables
  * $T$: iterations
- Space: $O(HMS \times N)$

Harmony Search turns optimization into an art form, 
solutions "play together" and gradually find the most resonant chord in the landscape of possibilities.

### 958. Firefly Algorithm

The Firefly Algorithm (FA) is a swarm intelligence method inspired by the flashing patterns of fireflies.
Each firefly represents a candidate solution that moves toward brighter (better) ones, with movement intensity governed by light absorption and random perturbation.

It's simple, parallel, and powerful for continuous, nonlinear, and multi-modal optimization problems.

#### What Problem Are We Solving?

Many optimization landscapes have multiple local minima.
Classic gradient-based algorithms easily get stuck in one.
Firefly Algorithm introduces *collective attraction* and *controlled randomness* to search globally while retaining local refinement.

It's especially effective for:

- Engineering optimization
- Image processing and feature selection
- Machine learning parameter tuning
- Benchmark function optimization

#### The Core Idea

Each firefly has a brightness proportional to its fitness (solution quality).
A less bright firefly moves toward a brighter one, and brightness decreases with distance.
Random motion ensures exploration when no brighter neighbor exists.

#### Mathematical Formulation

For fireflies $i$ and $j$ at positions $x_i$ and $x_j$, their movement is governed by:

1. Attractiveness function:
   $$
   \beta(r) = \beta_0 e^{-\gamma r^2}
   $$
   where

   * $\beta_0$ is the maximum attractiveness,
   * $\gamma$ is the light absorption coefficient,
   * $r = |x_i - x_j|$ is the distance.

2. Movement rule:
   $$
   x_i \leftarrow x_i + \beta_0 e^{-\gamma r^2} (x_j - x_i) + \alpha (\text{rand} - 0.5)
   $$

where:

- The second term moves $i$ toward brighter firefly $j$,
- The third term adds random noise controlled by $\alpha$.

#### How It Works (Plain Language)

Imagine fireflies on a dark field.
Each emits light according to how "good" its location is.
Dimmer fireflies fly toward brighter ones.
The closer they are, the stronger the attraction.
Over time, clusters of fireflies gather around the brightest spots, the best solutions.

Random motion keeps them from settling too early into suboptimal regions.

#### Step-by-Step Summary

| Step | Description                                                   |
| ---- | ------------------------------------------------------------- |
| 1    | Initialize random population of fireflies                     |
| 2    | Evaluate brightness using objective function                  |
| 3    | For each pair $(i,j)$, move $i$ toward $j$ if $j$ is brighter |
| 4    | Add small random motion                                       |
| 5    | Update brightness and repeat until convergence                |

#### Example

Goal: minimize $f(x) = x^2 + 3\sin(x)$

| Iteration | Firefly Positions  | Best Position     | Notes                  |
| --------- | ------------------ | ----------------- | ---------------------- |
| 1         | Random scattered   | 1.7               | Brightest (lowest $f$) |
| 5         | Cluster around 1.8 | Stable region     | Fewer random jumps     |
| 10        | All near 1.7       | Converged optimum |                        |

#### Tiny Code

Python

```python
import random, math

def f(x): return x2 + 3 * math.sin(x)

n = 10  # number of fireflies
beta0, gamma, alpha = 1.0, 0.5, 0.2
fireflies = [random.uniform(-5, 5) for _ in range(n)]

for _ in range(100):
    for i in range(n):
        for j in range(n):
            if f(fireflies[j]) < f(fireflies[i]):
                r = abs(fireflies[i] - fireflies[j])
                beta = beta0 * math.exp(-gamma * r2)
                fireflies[i] += beta * (fireflies[j] - fireflies[i]) + alpha * (random.random() - 0.5)

best = min(fireflies, key=f)
print("Best:", best, "f(x):", f(best))
```

C (Sketch)

```c
// Initialize positions x[i]
// For each iteration:
//   For each pair (i,j):
//     if f(x[j]) < f(x[i]): move x[i] toward x[j]
//     apply random perturbation
// Track best solution.
```

#### Why It Matters

- Global and local search balance, controlled by $\gamma$ and $\alpha$.
- Simple yet robust for continuous problems.
- Handles nonconvex, discontinuous, or noisy objectives.
- Adaptable to multiobjective and discrete variants.
- Naturally parallelizable, each firefly acts independently.

#### A Gentle Proof (Why It Works)

Attraction decays exponentially with distance, ensuring local convergence around good solutions while maintaining global diversity via random perturbations.

Formally, the expected displacement magnitude per iteration:

$$
E[|x_i^{t+1} - x_i^t|] \le \beta_0 e^{-\gamma r^2} |x_j - x_i| + O(\alpha)
$$

decreases as $\gamma$ increases or $\alpha$ decreases, guaranteeing stability once near minima.

#### Try It Yourself

1. Optimize $f(x) = x^2 + 10\sin(5x)$ and visualize movement.
2. Tune parameters:

   * $\alpha$ (randomness)
   * $\beta_0$ (attraction)
   * $\gamma$ (light absorption)
3. Compare with Particle Swarm Optimization.
4. Run in 2D and plot trajectories.
5. Try multi-modal functions like Rastrigin or Ackley.

#### Test Cases

| Problem            | Domain      | Notes                     |
| ------------------ | ----------- | ------------------------- |
| Sphere             | Continuous  | Fast convergence          |
| Rosenbrock         | Nonlinear   | Needs moderate randomness |
| Rastrigin          | Multi-modal | Global search test        |
| Engineering design | Continuous  | Real-world applicability  |

#### Complexity

- Time: $O(n^2)$ per iteration (pairwise attraction)
- Space: $O(n)$
- Convergence: controlled by $\alpha$, $\gamma$, and iteration limit

The Firefly Algorithm shows how collective attraction, simple, local, and luminous, can light the way to global optimization.

### 959. Bee Colony Optimization

Bee Colony Optimization (BCO) mimics how real bees forage for nectar, balancing exploration and exploitation through communication, recruitment, and local search.
Each bee represents a potential solution that explores or refines different regions of the search space. Over time, the colony collectively converges on the most promising "nectar sources", optimal or near-optimal solutions.

#### What Problem Are We Solving?

BCO is a population-based metaheuristic used for continuous and combinatorial optimization, especially where:

- The objective function is nonconvex or noisy
- Gradient information is unavailable
- Solutions must balance local refinement and global discovery

Applications include:

- Routing and scheduling
- Feature selection
- Engineering design
- Resource allocation

#### The Core Idea

A bee colony consists of:

1. Employed bees exploring known food sources (solutions)
2. Onlooker bees selecting food sources based on shared information
3. Scout bees randomly exploring new areas

Through cycles of communication and movement, the colony gradually refines toward optimal solutions.

#### Mathematical Formulation

Let:

- $x_i$ be the position (solution) of bee $i$
- $f(x_i)$ be the fitness (nectar quality)
- $N$ be the number of food sources

Each iteration involves:

1. Employed Bee Phase:
   Generate a neighbor solution for each $x_i$:
   $$
   v_{ij} = x_{ij} + \phi_{ij}(x_{ij} - x_{kj})
   $$
   where $k \ne i$ and $\phi_{ij} \in [-1, 1]$ is a random coefficient.

2. Fitness Evaluation:
   $$
   \text{fit}(x_i) = \frac{1}{1 + f(x_i)}
   $$

3. Onlooker Bee Phase:
   Choose food sources with probability proportional to fitness:
   $$
   p_i = \frac{\text{fit}(x_i)}{\sum_{j=1}^{N} \text{fit}(x_j)}
   $$

4. Scout Bee Phase:
   Replace stagnating solutions with random ones if they haven't improved for a preset limit:
   $$
   x_i = \text{random}(L, U)
   $$

5. Memorize the best solution.

#### How It Works (Plain Language)

Imagine a colony of bees searching for flowers.
Each bee first explores an area (solution) and shares its nectar amount (fitness) in the hive.
Onlooker bees watch and choose to follow successful bees to rich fields.
If a bee's source runs dry, it becomes a scout and searches anew.
Over time, bees cluster around the richest flowers, the global optimum.

#### Step-by-Step Summary

| Step | Description                       |
| ---- | --------------------------------- |
| 1    | Initialize random food sources    |
| 2    | Employed bees explore neighbors   |
| 3    | Onlookers choose best sources     |
| 4    | Scouts search new regions         |
| 5    | Memorize best solution and repeat |

#### Example

Goal: minimize $f(x) = x^2 + 4\sin(x)$

At each iteration:

- Bees explore around current best $x$
- Onlookers reinforce the promising region
- Scouts prevent stagnation by adding randomness

Eventually, bees gather near the global minimum around $x \approx -1.4$.

#### Tiny Code

Python

```python
import random, math

def f(x): return x2 + 4 * math.sin(x)

N, limit, max_iter = 10, 10, 100
L, U = -5, 5
bees = [random.uniform(L, U) for _ in range(N)]
trial = [0]*N

for _ in range(max_iter):
    for i in range(N):
        k = random.choice([j for j in range(N) if j != i])
        phi = random.uniform(-1, 1)
        v = bees[i] + phi * (bees[i] - bees[k])
        if f(v) < f(bees[i]):
            bees[i], trial[i] = v, 0
        else:
            trial[i] += 1
    prob = [1/(1+f(b)) for b in bees]
    s = sum(prob)
    prob = [p/s for p in prob]
    for i in range(N):
        if random.random() < prob[i]:
            k = random.choice([j for j in range(N) if j != i])
            phi = random.uniform(-1, 1)
            v = bees[i] + phi * (bees[i] - bees[k])
            if f(v) < f(bees[i]):
                bees[i], trial[i] = v, 0
    for i in range(N):
        if trial[i] > limit:
            bees[i] = random.uniform(L, U)
            trial[i] = 0

best = min(bees, key=f)
print("Best:", best, "f(x):", f(best))
```

#### Why It Matters

- Combines exploration (scouts) and exploitation (employed bees).
- Naturally parallelizable.
- Requires few parameters, only colony size and scout limit.
- Performs well in noisy, high-dimensional, and multi-modal spaces.
- Adapts dynamically through feedback between bees.

#### A Gentle Proof (Why It Works)

At each cycle, the probability of selecting higher-fitness solutions increases due to proportional recruitment.
The replacement of stagnant sources by scouts ensures continuous diversity.
Under repeated sampling, convergence toward global optima is probabilistically guaranteed as exploration persists.

#### Try It Yourself

1. Minimize $f(x, y) = x^2 + y^2$ with 20 bees.
2. Adjust colony size $N$ and scout limit.
3. Compare convergence with Particle Swarm Optimization and Firefly Algorithm.
4. Visualize swarm behavior across iterations.
5. Try discrete adaptation for Traveling Salesman Problem.

#### Test Cases

| Problem       | Domain     | Notes                     |
| ------------- | ---------- | ------------------------- |
| Sphere        | Continuous | Simple benchmark          |
| Ackley        | Nonlinear  | Tests exploration ability |
| Scheduling    | Discrete   | Suited for BCO variants   |
| Neural tuning | Continuous | Parallelizable            |

#### Complexity

- Time: $O(N \times \text{iterations})$
- Space: $O(N)$
- Convergence: Stochastic but stable for large populations

Bee Colony Optimization captures the power of cooperation, 
a swarm of simple agents, each with limited insight, finding order and intelligence through shared discovery.

### 960. Hill Climbing

Hill Climbing is one of the simplest optimization algorithms.
It mimics how a climber ascends a hill by always taking a step toward higher ground (better solutions).
Despite its simplicity, it's the foundation of many local search and metaheuristic algorithms.

#### What Problem Are We Solving?

Hill Climbing is used for optimization without gradients, when we can evaluate the quality of a solution but not compute its derivatives.
It's especially useful for:

- Discrete or combinatorial optimization
- Heuristic search problems
- Feature selection
- Parameter tuning

However, it may get stuck at local optima, points that seem best locally but are not globally optimal.

#### The Core Idea

At each step, Hill Climbing evaluates neighboring solutions and moves to the one with the best improvement.
If no neighbor is better, it stops, assuming it has reached a peak (local optimum).

#### Mathematical Formulation

Let $f(x)$ be the fitness or objective function we want to maximize.
We start with an initial solution $x_0$ and iterate:

1. Generate a neighbor $x'$ near $x_t$.
2. If $f(x') > f(x_t)$, move there:
   $$
   x_{t+1} = x'
   $$
   else stop.
3. Repeat until no improvement is found.

Formally:

$$
x_{t+1} =
\begin{cases}
x', & \text{if } f(x') > f(x_t),\\
x_t, & \text{otherwise.}
\end{cases}
$$


#### How It Works (Plain Language)

Imagine you are standing in fog on a hill.
You can't see far, but you can feel the slope.
You take small steps uphill, always toward increasing elevation.
Eventually, you stop when all nearby directions go downhill, you've reached a peak (though maybe not the highest one).

#### Step-by-Step Summary

| Step | Description                            |
| ---- | -------------------------------------- |
| 1    | Start with a random solution           |
| 2    | Generate a neighboring solution        |
| 3    | If neighbor is better, move there      |
| 4    | Repeat until no better neighbor exists |

Variants add randomness or restart from new positions to escape local optima.

#### Example

Maximize $f(x) = -x^2 + 5x$ over $x \in [0, 5]$

Start with $x = 1$
Neighbor step: $x' = x + 0.1$

Compute:
$$
f(x=1) = 4, \quad f(x'=1.1) = 4.39
$$
Move uphill.
Continue until slope becomes negative, near $x = 2.5$, where $f(x)$ peaks.

#### Tiny Code

Python

```python
import random

def f(x): return -x2 + 5*x

x = random.uniform(0, 5)
step = 0.1

for _ in range(100):
    x_new = x + random.choice([-step, step])
    if 0 <= x_new <= 5 and f(x_new) > f(x):
        x = x_new

print("Best x:", x, "f(x):", f(x))
```

C (Sketch)

```c
// Start with x = random value
// Loop: generate neighbor x_new
// If f(x_new) > f(x), move to x_new
// Stop when no better neighbor found
```

#### Why It Matters

- Foundation for local search and stochastic optimization.
- Works when gradient is unknown or non-differentiable.
- Forms the base for advanced methods:

  * Simulated Annealing (adds temperature-based randomness)
  * Tabu Search (adds memory)
  * Genetic Algorithms (adds population-based exploration)

#### A Gentle Proof (Why It Works)

If the search space is finite and each step improves $f(x)$, the algorithm must terminate at a local optimum since there's a finite number of states.
Convergence is guaranteed, but not necessarily to the global optimum.

Formally, since $f(x_{t+1}) > f(x_t)$ and $f$ is bounded above,
$$
\lim_{t \to \infty} (f(x_{t+1}) - f(x_t)) = 0
$$
thus Hill Climbing reaches a stable point.

#### Try It Yourself

1. Use Hill Climbing to maximize $f(x) = \sin(x)$ over $[0, 2\pi]$.
2. Add random restarts to escape local maxima.
3. Try stochastic Hill Climbing, sometimes accept worse solutions.
4. Compare to Simulated Annealing.
5. Visualize steps on a 2D surface like $f(x, y) = \sin(x)\cos(y)$.

#### Test Cases

| Function           | Domain      | Behavior                    |
| ------------------ | ----------- | --------------------------- |
| $-x^2 + 5x$        | Continuous  | Smooth single peak          |
| $\sin(x)$          | Multi-modal | Needs random restarts       |
| Rastrigin          | Multi-modal | Local optima test           |
| Traveling Salesman | Discrete    | Neighborhood swap heuristic |

#### Complexity

- Time: $O(T)$ (T = iterations)
- Space: $O(1)$
- Convergence: deterministic, local

Hill Climbing is the purest form of search, 
step by step, always upward, until no higher ground remains.
From this humble method, the modern landscape of metaheuristics begins.

# Section 97. Reinforcement Learning 

### 961. Monte Carlo Control

Monte Carlo Control is a cornerstone algorithm in reinforcement learning (RL) that estimates the optimal policy by sampling complete episodes and averaging the observed returns.
It learns from experience, no model of the environment required, by repeating trial and error across many episodes.

#### What Problem Are We Solving?

We want to learn how to act in an environment to maximize expected cumulative reward without knowing the underlying transition probabilities or reward model.

Given:

- A set of states $S$
- A set of actions $A$
- A reward function $R(s, a)$ (unknown)
- Discount factor $\gamma$

Monte Carlo Control learns:

- The action-value function $Q(s,a)$, expected return after taking $a$ in $s$ and following policy $\pi$
- The optimal policy $\pi^*(s) = \arg\max_a Q(s, a)$

#### The Core Idea

Monte Carlo methods estimate $Q(s,a)$ by running complete episodes and averaging returns that follow each $(s,a)$ pair.
Then, the policy is improved greedily with respect to the estimated $Q$.
Repeated alternation of evaluation and improvement converges to the optimal policy.

#### Mathematical Formulation

1. Return definition
   $$
   G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots
   $$

2. Action-value update
   $$
   Q(s,a) \leftarrow Q(s,a) + \alpha [G_t - Q(s,a)]
   $$
   where $G_t$ is the return following first occurrence of $(s,a)$.

3. Policy improvement
   $$
   \pi(s) = \arg\max_a Q(s,a)
   $$

4. Exploration control (ε-greedy):

$$
\pi(a|s) =
\begin{cases}
1 - \varepsilon + \dfrac{\varepsilon}{|A(s)|}, & \text{if } a = \arg\max_a Q(s,a),\\
\dfrac{\varepsilon}{|A(s)|}, & \text{otherwise.}
\end{cases}
$$


#### How It Works (Plain Language)

Think of a player learning a game (like blackjack) by playing it over and over:

- Every full game (episode) ends with a score (total reward).
- The player records what moves led to what outcomes.
- Over time, average results for each move form a reliable estimate of its true value.
- The player gradually shifts to moves that give higher average returns.

#### Step-by-Step Summary

| Step | Description                                          |
| ---- | ---------------------------------------------------- |
| 1    | Initialize $Q(s,a)$ arbitrarily                      |
| 2    | Initialize a policy $\pi$ (e.g., random or ε-greedy) |
| 3    | Generate an episode following $\pi$                  |
| 4    | Compute returns $G_t$ for all visited $(s,a)$        |
| 5    | Update $Q(s,a)$ by averaging returns                 |
| 6    | Improve $\pi$ greedily w.r.t. updated $Q$            |
| 7    | Repeat for many episodes                             |

#### Example

Suppose an agent plays blackjack.
Each round (episode), it records the sequence of state-action pairs and the final outcome (+1 win, -1 lose, 0 draw).
Over many rounds, it estimates the value of each $(s,a)$, e.g., "hit at 15" or "stand at 18", and adjusts its strategy accordingly.

#### Tiny Code

Python

```python
import random

# Environment (example): simplified blackjack
actions = ["hit", "stand"]
Q = {}
returns = {}

def policy(s, eps=0.1):
    if random.random() < eps:
        return random.choice(actions)
    return max(actions, key=lambda a: Q.get((s, a), 0))

for _ in range(10000):
    episode = []
    state = random.randint(4, 21)
    while True:
        a = policy(state)
        reward = random.choice([-1, 0, 1])  # placeholder for environment response
        episode.append((state, a, reward))
        if random.random() < 0.3:  # terminal
            break
        state = min(21, state + random.choice([-2, 1, 2]))
    G = 0
    for (s, a, r) in reversed(episode):
        G = r + 0.9 * G
        if not any(x[0] == s and x[1] == a for x in episode[:-1]):
            returns.setdefault((s,a), []).append(G)
            Q[(s,a)] = sum(returns[(s,a)]) / len(returns[(s,a)])
```

#### Why It Matters

- Model-free: learns directly from experience without needing transition probabilities.
- Simple but powerful: the basis for many RL methods like SARSA, Q-learning, and Actor–Critic.
- Exploration-friendly: ε-greedy ensures all actions are tried enough times.
- Theoretical guarantee: with infinite exploration, $Q \to Q^*$.

#### A Gentle Proof (Why It Works)

By the Law of Large Numbers, the empirical average of returns for $(s,a)$ converges to its expected value under policy $\pi$:
$$
E[G_t | S_t = s, A_t = a] = q_\pi(s,a)
$$
Successive policy improvement ensures:
$$
q_{\pi_{k+1}}(s,a) \ge q_{\pi_k}(s,a)
$$
Thus, repeated alternation of evaluation and improvement converges to $\pi^*$.

#### Try It Yourself

1. Implement Monte Carlo control for a gridworld or blackjack.
2. Compare first-visit vs every-visit Monte Carlo updates.
3. Visualize how $Q(s,a)$ stabilizes over episodes.
4. Experiment with different $\varepsilon$ for exploration.
5. Try decaying $\varepsilon_t$ for late-stage exploitation.

#### Test Cases

| Environment     | Description      | Reward Signal             |
| --------------- | ---------------- | ------------------------- |
| Blackjack       | Episodic         | +1 win, -1 lose           |
| Gridworld       | Finite horizon   | Step cost and goal reward |
| Tic-Tac-Toe     | Discrete actions | +1 win, -1 lose, 0 draw   |
| Maze navigation | Continuous       | Sparse terminal reward    |

#### Complexity

- Time: $O(N \times T)$ per episode (N states, T steps per episode)
- Space: $O(|S||A|)$
- Convergence: guaranteed with infinite sampling under $\varepsilon$-greedy policy

Monte Carlo Control embodies the spirit of learning from experience, 
no models, no equations of motion, just repeated play and careful averaging until knowledge emerges.

### 962. Temporal Difference (TD) Learning

Temporal Difference (TD) Learning is a reinforcement learning algorithm that blends ideas from Monte Carlo methods and Dynamic Programming.
It learns directly from experience, step by step, by updating value estimates using predictions of future estimates rather than waiting for full episodes to finish.

#### What Problem Are We Solving?

Monte Carlo control waits until the end of each episode to update its value estimates.
This can be slow or infeasible in continuing tasks.

TD learning solves this by *bootstrapping*, updating estimates based partly on existing estimates:

- Learn state values ($V(s)$) or action values ($Q(s,a)$)
- Without knowing the environment model
- During (not after) episodes

It's used in:

- Game playing (e.g., TD-Gammon)
- Online prediction
- Robot control
- Financial forecasting

#### The Core Idea

TD learning updates a value function based on the difference between:

- The predicted return at one time step, and
- The better-informed prediction at the next step.

That difference is called the temporal difference error.

#### Mathematical Formulation

1. Value update rule (TD(0)):
   $$
   V(S_t) \leftarrow V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]
   $$
   where

   * $\alpha$ is the learning rate
   * $\gamma$ is the discount factor

2. The term
   $$
   \delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)
   $$
   is the TD error, how surprising the next observation is.

3. The same idea extends to Q-values:
   $$
   Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]
   $$

#### How It Works (Plain Language)

Imagine predicting a movie's rating as you watch it.
After every scene, you adjust your expectation based on how the movie is going so far, without waiting for the credits to roll.

That's TD learning: you refine your predictions as new evidence arrives, using your *current predictions* as stepping stones.

#### Step-by-Step Summary

| Step | Description                                       |
| ---- | ------------------------------------------------- |
| 1    | Initialize $V(s)$ or $Q(s,a)$ arbitrarily         |
| 2    | Start an episode with state $S_0$                 |
| 3    | Choose action $A_t$ (ε-greedy)                    |
| 4    | Observe reward $R_{t+1}$ and next state $S_{t+1}$ |
| 5    | Compute TD error $\delta_t$                       |
| 6    | Update $V(S_t)$ or $Q(S_t, A_t)$                  |
| 7    | Repeat for all time steps                         |

#### Example

Let $V(A)=0.5$, $\gamma=0.9$, $\alpha=0.1$
You move from state A to B, get reward $R=1$, and $V(B)=0.6$.

Compute:
$$
\delta = R + \gamma V(B) - V(A) = 1 + 0.9 \times 0.6 - 0.5 = 1.04
$$
Update:
$$
V(A) \leftarrow 0.5 + 0.1 \times 1.04 = 0.604
$$

#### Tiny Code

Python

```python
import random

V = {s: 0.0 for s in range(5)}
alpha, gamma = 0.1, 0.9

for episode in range(100):
    s = random.choice(list(V.keys()))
    for _ in range(5):
        next_s = random.choice(list(V.keys()))
        r = random.uniform(-1, 1)
        V[s] += alpha * (r + gamma * V[next_s] - V[s])
        s = next_s

print(V)
```

C (Sketch)

```c
// For each step:
//   delta = R + gamma * V[next_s] - V[s];
//   V[s] += alpha * delta;
```

#### Why It Matters

- Online learning, updates happen as experience unfolds.
- Efficient, no need to store full episodes.
- Bootstrapping, learns from both experience and its own predictions.
- Foundation for advanced algorithms like SARSA, Q-learning, and Actor–Critic.

#### A Gentle Proof (Why It Works)

By the contraction property of Bellman operators, repeated TD updates converge to the true value function $V_\pi(s)$ under a fixed policy $\pi$.
Each update moves $V(s)$ closer to the expected discounted return:
$$
E[V(S_t)] \to v_\pi(s)
$$
As long as every state is visited infinitely often and $\alpha_t$ satisfies standard conditions ($\sum \alpha_t = \infty$, $\sum \alpha_t^2 < \infty$).

#### Try It Yourself

1. Implement TD(0) for a random-walk environment.
2. Compare learning curves for different $\alpha$.
3. Visualize how $V(s)$ converges to true values.
4. Extend to TD(λ), multi-step backup averaging.
5. Compare with Monte Carlo estimates.

#### Test Cases

| Environment | Description           | Notes                       |
| ----------- | --------------------- | --------------------------- |
| Random Walk | Classic example       | Smooth convergence          |
| Gridworld   | State transitions     | Online update visualization |
| Tic-Tac-Toe | Predict game outcomes | Model-free                  |
| Robot path  | Continuous control    | Real-time learning          |

#### Complexity

- Time: $O(T)$ per episode (T = steps)
- Space: $O(|S|)$ or $O(|S||A|)$
- Convergence: guaranteed for small $\alpha$ under on-policy exploration

Temporal Difference Learning teaches the power of bootstrapped prediction, 
learning not by looking back after everything is over, but by predicting, updating, and improving as you go.

### 963. SARSA (On-Policy Temporal Difference Learning)

SARSA (State–Action–Reward–State–Action) is a classic on-policy reinforcement learning algorithm.
It extends Temporal Difference (TD) learning to directly estimate action values $Q(s,a)$ rather than just state values.
The name comes from the five elements used in each update: $(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$.

#### What Problem Are We Solving?

We want to learn the optimal policy that maximizes long-term reward, but unlike Monte Carlo methods, we don't need to wait for the episode to finish.
Unlike Q-learning, SARSA updates using the action actually taken by the current policy, not the greedy one, making it "on-policy."

It's used for:

- Online control and navigation tasks
- Robot and autonomous vehicle control
- Any environment where exploration must be safe or gradual

#### The Core Idea

SARSA updates the value of the current state-action pair using:

- The immediate reward $R_{t+1}$
- The predicted value of the *next* state-action pair under the current policy

It learns while following the policy it's evaluating, balancing exploration (via ε-greedy) and exploitation (choosing best actions).

#### Mathematical Formulation

1. TD update rule:
   $$
   Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]
   $$

2. Action selection (ε-greedy):

$$
A_t =
\begin{cases}
\arg\max_a Q(S_t, a), & \text{with probability } 1 - \varepsilon,\\
\text{random action}, & \text{with probability } \varepsilon.
\end{cases}
$$


#### How It Works (Plain Language)

Imagine teaching a robot to walk.
Each move changes its position (state) and gives feedback (reward).
The robot updates its understanding of which move was good or bad, based on what actually happened, not on some hypothetical "best" move it didn't take.
That's SARSA: learning *from experience following your own policy*.

#### Step-by-Step Summary

| Step | Description                                               |
| ---- | --------------------------------------------------------- |
| 1    | Initialize $Q(s,a)$ arbitrarily                           |
| 2    | Start at initial state $S_0$                              |
| 3    | Choose $A_0$ from $S_0$ using ε-greedy                    |
| 4    | Repeat for each step:                                     |
|      | a. Take $A_t$, observe $R_{t+1}$ and $S_{t+1}$            |
|      | b. Choose next action $A_{t+1}$ (ε-greedy)                |
|      | c. Update $Q(S_t, A_t)$ using TD rule                     |
|      | d. Set $S_t \leftarrow S_{t+1}$, $A_t \leftarrow A_{t+1}$ |

#### Example

Suppose $Q(s,a)$ is initialized to zero.
At time $t$:

- $S_t = s_1$, $A_t = \text{right}$, reward $R_{t+1} = 1$
- Next state $S_{t+1} = s_2$, action $A_{t+1} = \text{up}$
- $\alpha = 0.1$, $\gamma = 0.9$

Then:
$$
Q(s_1, \text{right}) \leftarrow 0 + 0.1 [1 + 0.9 Q(s_2, \text{up}) - 0] = 0.1
$$

#### Tiny Code

Python

```python
import random

Q = {}
states = ["A", "B", "C"]
actions = ["left", "right"]
alpha, gamma, eps = 0.1, 0.9, 0.1

def policy(s):
    if random.random() < eps:
        return random.choice(actions)
    return max(actions, key=lambda a: Q.get((s,a), 0))

for episode in range(1000):
    s = random.choice(states)
    a = policy(s)
    for _ in range(10):
        r = random.uniform(-1, 1)
        s_next = random.choice(states)
        a_next = policy(s_next)
        q = Q.get((s,a), 0)
        q_next = Q.get((s_next,a_next), 0)
        Q[(s,a)] = q + alpha * (r + gamma * q_next - q)
        s, a = s_next, a_next
```

#### Why It Matters

- On-policy control, learns safely using the same policy it acts with.
- Smoothly transitions from exploration to exploitation.
- Proven convergence under decaying $\varepsilon$ and $\alpha$.
- Serves as a foundation for Expected SARSA, n-step SARSA, and TD(λ).

#### A Gentle Proof (Why It Works)

For a fixed policy $\pi$, the SARSA update approximates the Bellman equation:
$$
Q_\pi(S_t, A_t) = E[R_{t+1} + \gamma Q_\pi(S_{t+1}, A_{t+1})]
$$
Each update step is a stochastic approximation to this expectation.
Given sufficient exploration and diminishing learning rate, $Q \to Q_\pi$, and greedy improvement leads to $\pi^*$.

#### Try It Yourself

1. Implement SARSA in a gridworld (like OpenAI Gym's CliffWalking).
2. Compare performance with Q-learning (off-policy).
3. Test different ε values to see exploration effects.
4. Try n-step SARSA for faster convergence.
5. Visualize learned $Q(s,a)$ heatmaps.

#### Test Cases

| Environment  | Description            | Notes                         |
| ------------ | ---------------------- | ----------------------------- |
| CliffWalking | Classic on-policy test | Avoids cliff with safe policy |
| Gridworld    | Deterministic          | Good for visualization        |
| Taxi-v3      | Discrete navigation    | Requires exploration          |
| FrozenLake   | Stochastic             | Highlights ε-greedy balance   |

#### Complexity

- Time: $O(|S||A|)$ per update step
- Space: $O(|S||A|)$
- Convergence: guaranteed under GLIE (greedy in the limit with infinite exploration)

SARSA shows the essence of learning while doing, 
each decision refines the policy that made it, blending action and reflection into one continuous loop of improvement.

### 964. Q-Learning (Off-Policy Temporal Difference Control)

Q-Learning is one of the most influential reinforcement learning algorithms.
It learns the optimal action-value function directly, even while following a different (exploratory) behavior policy.
Unlike SARSA, which learns from the actions it *actually* takes, Q-Learning learns from the *best possible* actions it *could* take, making it an off-policy method.

#### What Problem Are We Solving?

We want to find the optimal policy $\pi^*(s)$ that maximizes cumulative reward in an unknown environment.
Q-Learning does this without requiring a model of the environment's dynamics.

It is used in:

- Games (e.g., AlphaGo's early versions)
- Navigation and control
- Autonomous decision-making systems
- Continuous adaptation tasks

#### The Core Idea

At each step, Q-Learning updates the estimate of $Q(s,a)$ toward the *best possible* next action's value, not necessarily the one taken by the current policy.

This makes Q-Learning off-policy:

- The behavior policy (exploration) decides what to do.
- The target policy (greedy) decides what the learner aims for.

#### Mathematical Formulation

1. Update rule:
   $$
   Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \big[ R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t) \big]
   $$

2. Greedy target:
   The term $\max_{a'} Q(S_{t+1}, a')$ represents the best estimated future return from the next state.

3. Action selection (ε-greedy):

$$
A_t =
\begin{cases}
\arg\max_a Q(S_t, a), & \text{with probability } 1 - \varepsilon,\\
\text{random action}, & \text{with probability } \varepsilon.
\end{cases}
$$


#### How It Works (Plain Language)

Think of Q-Learning as learning the *map of rewards* in an environment.
Every time you move, you update what you believe the best future payoff is, not just from what you did, but from what you *could have done better*.
Over many steps, this map converges to the true optimal reward landscape.

#### Step-by-Step Summary

| Step | Description                                               |
| ---- | --------------------------------------------------------- |
| 1    | Initialize $Q(s,a)$ arbitrarily                           |
| 2    | For each episode:                                         |
|      | a. Start from initial state $S_0$                         |
|      | b. Choose action $A_t$ using ε-greedy                     |
|      | c. Take action, observe $R_{t+1}$ and $S_{t+1}$           |
|      | d. Update $Q(S_t, A_t)$ using TD rule                     |
|      | e. Set $S_t \leftarrow S_{t+1}$ and repeat until terminal |

#### Example

Suppose:

- $S_t = s_1$, $A_t = \text{right}$
- Reward $R_{t+1} = 1$, next state $S_{t+1} = s_2$
- $\max_a Q(s_2, a) = 4$
- $\alpha = 0.1$, $\gamma = 0.9$, $Q(s_1, \text{right}) = 2$

Then:
$$
Q(s_1, \text{right}) \leftarrow 2 + 0.1 [1 + 0.9 \times 4 - 2] = 2 + 0.1 \times 3.6 = 2.36
$$

#### Tiny Code

Python

```python
import random

Q = {}
states = ["A", "B", "C"]
actions = ["left", "right"]
alpha, gamma, eps = 0.1, 0.9, 0.1

def policy(s):
    if random.random() < eps:
        return random.choice(actions)
    return max(actions, key=lambda a: Q.get((s,a), 0))

for episode in range(1000):
    s = random.choice(states)
    while True:
        a = policy(s)
        r = random.uniform(-1, 1)
        s_next = random.choice(states)
        q = Q.get((s,a), 0)
        q_next_max = max([Q.get((s_next,a2), 0) for a2 in actions])
        Q[(s,a)] = q + alpha * (r + gamma * q_next_max - q)
        s = s_next
        if random.random() < 0.2:  # terminate
            break
```

#### Why It Matters

- Model-free, learns directly from experience.
- Off-policy, can learn optimal policy while exploring randomly.
- Guaranteed convergence to $Q^*$ under certain conditions (Watkins & Dayan, 1992).
- Forms the foundation for Deep Q-Networks (DQN) and modern deep RL.

#### A Gentle Proof (Why It Works)

For a deterministic learning rate $\alpha_t$ and sufficient exploration, Q-Learning approximates the Bellman optimality equation:
$$
Q^*(s,a) = E[R_{t+1} + \gamma \max_{a'} Q^*(S_{t+1}, a')]
$$
Since the Bellman operator is a contraction mapping, repeated application guarantees convergence:
$$
| Q_{t+1} - Q^* |*\infty \le \gamma | Q_t - Q^* |*\infty
$$
Hence $Q_t \to Q^*$ as $t \to \infty$.

#### Try It Yourself

1. Implement Q-Learning for the FrozenLake environment.
2. Compare learning curves with SARSA.
3. Tune $\varepsilon$ for exploration and convergence speed.
4. Add a decaying $\alpha_t$ to stabilize learning.
5. Visualize the learned $Q$-table or policy heatmap.

#### Test Cases

| Environment  | Description   | Notes                             |
| ------------ | ------------- | --------------------------------- |
| FrozenLake   | Discrete grid | Off-policy advantage              |
| Taxi-v3      | Navigation    | Faster convergence than SARSA     |
| CliffWalking | Risky terrain | SARSA safer, Q-learning bolder    |
| Gridworld    | Small testbed | Perfect for Q-table visualization |

#### Complexity

- Time: $O(|S||A|)$ per update
- Space: $O(|S||A|)$
- Convergence: guaranteed under GLIE and Robbins–Monro learning rate conditions

Q-Learning is the heart of modern reinforcement learning, 
a learner that imagines better futures and improves itself toward them, one step at a time.

### 965. Double Q-Learning

Double Q-Learning refines standard Q-Learning by solving one of its biggest weaknesses: overestimation bias.
In regular Q-Learning, the same values are used both to select and to evaluate actions, this tends to overrate some actions, especially in noisy environments.
Double Q-Learning fixes this by maintaining two separate value estimators, which keep each other honest.

#### What Problem Are We Solving?

Standard Q-Learning uses the same $Q$ function to both:

1. Choose the best next action via $\max_a Q(S', a)$
2. Evaluate that chosen action's value

This self-referential step can produce optimistic overestimates, which make learning unstable or slow.

Double Q-Learning reduces that bias by decoupling *action selection* and *action evaluation.*

#### The Core Idea

Use two value functions, $Q^A$ and $Q^B$, that take turns learning from each other.

- One function ($Q^A$) chooses the next action (selection)
- The other ($Q^B$) evaluates that action (evaluation)

By alternating updates, the system learns more accurate and stable estimates of true $Q^*$.

#### Mathematical Formulation

1. Maintain two estimators, $Q^A$ and $Q^B$
2. With equal probability, update one of them at each step:

If updating $Q^A$:
$$
Q^A(S_t, A_t) \leftarrow Q^A(S_t, A_t) + \alpha [R_{t+1} + \gamma Q^B(S_{t+1}, \arg\max_a Q^A(S_{t+1}, a)) - Q^A(S_t, A_t)]
$$

If updating $Q^B$:
$$
Q^B(S_t, A_t) \leftarrow Q^B(S_t, A_t) + \alpha [R_{t+1} + \gamma Q^A(S_{t+1}, \arg\max_a Q^B(S_{t+1}, a)) - Q^B(S_t, A_t)]
$$

#### How It Works (Plain Language)

Imagine two friends, Alice and Bob, both trying to estimate how good each action is.
Alice picks the best-looking action according to *her* table, but Bob gives the score.
Then next time, Bob picks and Alice scores.
Because each one checks the other's optimism, their shared knowledge becomes more reliable.

#### Step-by-Step Summary

| Step | Description                                               |
| ---- | --------------------------------------------------------- |
| 1    | Initialize $Q^A$ and $Q^B$ arbitrarily                    |
| 2    | For each episode:                                         |
|      | a. Choose $A_t$ using ε-greedy over $(Q^A + Q^B)$         |
|      | b. Take action $A_t$, observe $R_{t+1}$ and $S_{t+1}$     |
|      | c. Randomly choose which $Q$ to update                    |
|      | d. Use one for action selection, the other for evaluation |
|      | e. Repeat until episode ends                              |

#### Example

Suppose:

- $S_t = s_1$, $A_t = \text{right}$
- $R_{t+1} = 2$, $S_{t+1} = s_2$
- $\gamma = 0.9$, $\alpha = 0.1$
- $Q^A(s_1,\text{right}) = 1.5$
- $Q^A(s_2,\text{up}) = 2.0$, $Q^B(s_2,\text{up}) = 1.8$

If we update $Q^A$:
$$
Q^A(s_1,\text{right}) \leftarrow 1.5 + 0.1 [2 + 0.9 \times Q^B(s_2, \arg\max_a Q^A(s_2,a)) - 1.5]
$$
Since $\arg\max_a Q^A(s_2,a)$ is "up,"
$$
Q^A(s_1,\text{right}) = 1.5 + 0.1 [2 + 0.9 \times 1.8 - 1.5] = 1.5 + 0.1 \times 2.12 = 1.712
$$

#### Tiny Code

Python

```python
import random

Q_A, Q_B = {}, {}
states = ["A", "B", "C"]
actions = ["left", "right"]
alpha, gamma, eps = 0.1, 0.9, 0.1

def policy(s):
    Q_total = {a: Q_A.get((s,a),0) + Q_B.get((s,a),0) for a in actions}
    if random.random() < eps:
        return random.choice(actions)
    return max(actions, key=lambda a: Q_total[a])

for episode in range(1000):
    s = random.choice(states)
    while True:
        a = policy(s)
        r = random.uniform(-1, 1)
        s_next = random.choice(states)
        if random.random() < 0.5:
            a_next = max(actions, key=lambda a: Q_A.get((s_next,a),0))
            Q_A[(s,a)] = Q_A.get((s,a),0) + alpha * (
                r + gamma * Q_B.get((s_next,a_next),0) - Q_A.get((s,a),0))
        else:
            a_next = max(actions, key=lambda a: Q_B.get((s_next,a),0))
            Q_B[(s,a)] = Q_B.get((s,a),0) + alpha * (
                r + gamma * Q_A.get((s_next,a_next),0) - Q_B.get((s,a),0))
        s = s_next
        if random.random() < 0.2:
            break
```

#### Why It Matters

- Reduces overestimation, more stable learning curves than Q-Learning
- Converges more smoothly in stochastic environments
- Foundation for Double DQN, a deep variant widely used in modern RL
- Encourages better value calibration in uncertain domains

#### A Gentle Proof (Why It Works)

In Q-Learning, $\max_a Q(S',a)$ tends to overestimate because of sampling noise.
Double Q-Learning breaks that coupling:
$$
E[Q^B(S', \arg\max_a Q^A(S',a))] \le E[\max_a Q^A(S',a)]
$$
Thus, bias is reduced while retaining consistency with the Bellman optimality equation.
The method still converges to $Q^*$ under standard conditions.

#### Try It Yourself

1. Run both Q-Learning and Double Q-Learning on the same environment.
2. Plot value estimates, notice Q-Learning's optimistic bias.
3. Tune learning rate $\alpha$ and discount $\gamma$.
4. Extend to Double DQN using neural networks.
5. Try stochastic reward functions to highlight bias effects.

#### Test Cases

| Environment  | Description            | Observation                        |
| ------------ | ---------------------- | ---------------------------------- |
| CliffWalking | Noisy terrain          | More stable than Q-Learning        |
| FrozenLake   | Stochastic transitions | Less variance                      |
| Taxi-v3      | Sparse rewards         | Smoother convergence               |
| Gridworld    | Simple grid            | Easier visualization of Q_A vs Q_B |

#### Complexity

- Time: $O(2|S||A|)$ (dual Q-tables)
- Space: $O(2|S||A|)$
- Convergence: unbiased, stable for small $\alpha$ and persistent exploration

Double Q-Learning is like learning from a partner, 
each half of the system cross-checks the other, turning optimism into balance and convergence into confidence.

### 966. Deep Q-Network (DQN)

Deep Q-Network (DQN) extends classical Q-Learning by using a neural network to approximate the action-value function $Q(s, a)$.
Instead of storing a table for every state-action pair, DQN generalizes across large or continuous state spaces, enabling reinforcement learning to work on high-dimensional inputs like images, games, and sensor data.

#### What Problem Are We Solving?

Traditional Q-Learning breaks down when:

- The state space is too large to store $Q(s,a)$ explicitly
- States are continuous (e.g., robot positions, pixels)
- Function approximation is needed

DQN addresses this by learning $Q_\theta(s, a)$, a parameterized function approximated by a deep neural network with weights $\theta$.

Applications include:

- Atari game playing (DeepMind, 2015)
- Autonomous control systems
- Decision-making from raw sensory inputs

#### The Core Idea

Approximate the optimal Q-function $Q^*(s,a)$ with a neural network:
$$
Q(s,a;\theta) \approx Q^*(s,a)
$$

Then apply the Q-Learning update rule, but instead of updating a single entry in a table, minimize the mean-squared error between the current prediction and the target:
$$
L(\theta) = \mathbb{E}\Big[ \big( y - Q(s,a;\theta) \big)^2 \Big]
$$
where the target value is:
$$
y = R + \gamma \max_{a'} Q(s',a';\theta^-)
$$

$\theta^-$ are the parameters of a target network, updated periodically for stability.

#### How It Works (Plain Language)

DQN learns how good each action is by *predicting future rewards* with a neural network.
It observes transitions $(s, a, r, s')$, stores them, and samples random mini-batches to train.
This "experience replay" helps it avoid overfitting to recent experiences and stabilize learning.

#### Step-by-Step Summary

| Step | Description                                                             |
| ---- | ----------------------------------------------------------------------- |
| 1    | Initialize replay buffer $D$                                            |
| 2    | Initialize Q-network with random weights $\theta$                       |
| 3    | Initialize target network with weights $\theta^- = \theta$              |
| 4    | For each episode:                                                       |
|      | a. Observe state $s$, choose action $a$ via ε-greedy on $Q(s,a;\theta)$ |
|      | b. Execute $a$, observe reward $r$ and next state $s'$                  |
|      | c. Store $(s,a,r,s')$ in replay buffer                                  |
|      | d. Sample random batch from $D$                                         |
|      | e. Compute target $y = r + \gamma \max_{a'} Q(s',a';\theta^-)$          |
|      | f. Minimize loss $(y - Q(s,a;\theta))^2$                                |
|      | g. Periodically update $\theta^- \leftarrow \theta$                     |

#### Mathematical Formulation

1. Loss function
   $$
   L(\theta) = \mathbb{E}*{(s,a,r,s') \sim D} \Big[ \big( R + \gamma \max*{a'} Q(s',a';\theta^-) - Q(s,a;\theta) \big)^2 \Big]
   $$

2. Gradient descent update
   $$
   \theta \leftarrow \theta - \alpha \nabla_\theta L(\theta)
   $$

3. Target network update
   $$
   \theta^- \leftarrow \theta \text{ every } C \text{ steps.}
   $$

#### Example

In Atari "Breakout":

- Input: 84×84 grayscale game frames
- Output: predicted Q-values for 4 possible actions
- Reward: +1 when the ball breaks a brick, 0 otherwise
  The network learns to move the paddle optimally to maximize cumulative score.

#### Tiny Code

Python (simplified DQN skeleton)

```python
import torch, torch.nn as nn, torch.optim as optim, random
import numpy as np

class DQN(nn.Module):
    def __init__(self, n_states, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, 128), nn.ReLU(),
            nn.Linear(128, n_actions)
        )
    def forward(self, x):
        return self.net(x)

q_net = DQN(4, 2)
target_net = DQN(4, 2)
target_net.load_state_dict(q_net.state_dict())

optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
gamma = 0.99

def update(batch):
    s, a, r, s_next = batch
    q_values = q_net(torch.tensor(s, dtype=torch.float32))
    next_q = target_net(torch.tensor(s_next, dtype=torch.float32)).max(1)[0].detach()
    y = torch.tensor(r) + gamma * next_q
    loss = ((q_values.gather(1, torch.tensor(a).unsqueeze(1)).squeeze() - y)2).mean()
    optimizer.zero_grad(); loss.backward(); optimizer.step()
```

#### Why It Matters

- Scales to high-dimensional inputs (e.g., images, sensors).
- Experience replay breaks correlation between consecutive samples.
- Target network prevents feedback loops and instability.
- Major milestone: Atari games achieved superhuman performance using only pixels as input.

#### A Gentle Proof (Why It Works)

The Bellman optimality operator:
$$
\mathcal{T}Q(s,a) = \mathbb{E}[R + \gamma \max_{a'} Q(s',a')]
$$
is a contraction mapping in $|\cdot|_\infty$.
By minimizing $L(\theta)$, DQN seeks to approximate the fixed point $Q^*$ of $\mathcal{T}$.
Target networks and replay buffers stabilize this iterative approximation, ensuring convergence in expectation.

#### Try It Yourself

1. Implement DQN for CartPole-v1 (OpenAI Gym).
2. Try different architectures, linear vs convolutional.
3. Compare with tabular Q-Learning performance.
4. Tune buffer size, batch size, and target update rate.
5. Visualize training reward over time.

#### Test Cases

| Environment    | Description       | Notes                        |
| -------------- | ----------------- | ---------------------------- |
| CartPole-v1    | Classic benchmark | Quick convergence            |
| MountainCar-v0 | Sparse rewards    | Needs exploration            |
| Atari Breakout | Visual input      | Deep CNN required            |
| LunarLander-v2 | Continuous state  | Sensitive to hyperparameters |

#### Complexity

- Time: $O(B \times E)$ (batch size × episodes)
- Space: $O(|D|)$ (replay buffer size)
- Convergence: approximate; improved with Double DQN and Dueling DQN

DQN marked the birth of Deep Reinforcement Learning, 
a fusion of neural representation and temporal learning that allows machines to learn complex behaviors directly from pixels and experience.

### 967. REINFORCE (Policy Gradient by Sampling)

REINFORCE is one of the simplest and most fundamental policy gradient algorithms.
Instead of learning a value function like Q-Learning, it directly learns the policy, that is, how to act.
By adjusting its parameters to increase the probability of rewarding actions, REINFORCE captures the essence of learning from experience.

#### What Problem Are We Solving?

Value-based methods (like Q-Learning or DQN) approximate $Q(s,a)$ and act greedily.
But in many problems:

- The action space is continuous
- Deterministic policies perform poorly
- Stochastic exploration is essential

REINFORCE solves this by optimizing a parameterized stochastic policy $\pi_\theta(a|s)$, without needing to learn $Q(s,a)$.

It is ideal for:

- Continuous control (robotics, game agents)
- Policy optimization in environments with stochastic transitions
- Learning with parameterized actions (e.g., probabilities, torques, velocities)

#### The Core Idea

Maximize the expected cumulative reward:
$$
J(\theta) = \mathbb{E}*{\pi*\theta}[R]
$$

Using the log-derivative trick:
$$
\nabla_\theta J(\theta) = \mathbb{E}*{\pi*\theta}\big[ \nabla_\theta \log \pi_\theta(a|s) , G_t \big]
$$

where $G_t$ is the return (total discounted reward from time $t$).

The update rule becomes:
$$
\theta \leftarrow \theta + \alpha , G_t , \nabla_\theta \log \pi_\theta(a_t|s_t)
$$

This moves the policy parameters $\theta$ in the direction that increases the log-probability of actions that led to higher rewards.

#### How It Works (Plain Language)

Imagine a player who tries different strategies, receives scores, and remembers which choices led to better outcomes.
REINFORCE simply nudges the probabilities of those actions to make them more likely in the future.
Over time, the agent learns which actions tend to yield higher rewards, directly shaping its behavior.

#### Step-by-Step Summary

| Step | Description                                                                                        |                                                      |
| ---- | -------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| 1    | Initialize policy parameters $\theta$ randomly                                                     |                                                      |
| 2    | For each episode:                                                                                  |                                                      |
|      | a. Run the policy $\pi_\theta(a                                                                    | s)$ to generate a trajectory $(s_0,a_0,r_1,s_1,...)$ |
|      | b. Compute the return $G_t = r_{t+1} + \gamma r_{t+2} + \dots$ for each time step                  |                                                      |
|      | c. Update parameters: $\theta \leftarrow \theta + \alpha , G_t , \nabla_\theta \log \pi_\theta(a_t | s_t)$                                                |
|      | d. Repeat until convergence                                                                        |                                                      |

#### Mathematical Formulation

1. Expected return
   $$
   J(\theta) = \mathbb{E}*{\pi*\theta}\Big[ \sum_{t=0}^T \gamma^t R_t \Big]
   $$

2. Policy gradient theorem
   $$
   \nabla_\theta J(\theta) = \mathbb{E}*{\pi*\theta}\Big[ \nabla_\theta \log \pi_\theta(a_t|s_t) (G_t - b_t) \Big]
   $$

3. Baseline term $b_t$
   Subtracting a baseline (often a value estimate) reduces variance without changing the expectation.

#### Example

Suppose:

- $\pi_\theta(a|s)$ is a softmax policy:
  $$
  \pi_\theta(a|s) = \frac{e^{\theta^\top f(s,a)}}{\sum_b e^{\theta^\top f(s,b)}}
  $$

- Reward $R_t = +1$ for good actions, $-1$ for bad.
  Then REINFORCE increases $\theta$ for actions with positive $G_t$, and decreases it otherwise.

#### Tiny Code

Python (with softmax policy)

```python
import numpy as np

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

theta = np.random.randn(2)  # parameter vector
alpha, gamma = 0.01, 0.99

for episode in range(1000):
    states, actions, rewards = [], [], []
    s = 0
    for t in range(10):
        probs = softmax(theta)
        a = np.random.choice(len(probs), p=probs)
        r = 1 if a == 1 else -1  # example reward
        states.append(s); actions.append(a); rewards.append(r)
    G = 0
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        grad_log = np.zeros_like(theta)
        grad_log[actions[t]] = 1 - softmax(theta)[actions[t]]
        theta += alpha * G * grad_log
```

#### Why It Matters

- Direct policy optimization, no Q-tables needed
- Supports continuous actions (unlike Q-learning)
- Simple and general, basis for all policy gradient methods
- Works with neural networks (e.g., in actor–critic models)

#### A Gentle Proof (Why It Works)

Using the log-derivative identity:
$$
\nabla_\theta \pi_\theta(a|s) = \pi_\theta(a|s) \nabla_\theta \log \pi_\theta(a|s)
$$

Then:
$$
\nabla_\theta J(\theta) = \sum_s d_\pi(s) \sum_a Q^\pi(s,a) \nabla_\theta \log \pi_\theta(a|s)
$$

The gradient is unbiased but high-variance; the baseline $b_t$ term reduces this variance without biasing the result.

#### Try It Yourself

1. Implement REINFORCE for CartPole-v1.
2. Compare with Actor–Critic methods, notice variance differences.
3. Add a value function baseline for variance reduction.
4. Use a neural network for $\pi_\theta(a|s)$.
5. Plot the total reward vs. episode count.

#### Test Cases

| Environment           | Description               | Observation                   |
| --------------------- | ------------------------- | ----------------------------- |
| CartPole-v1           | Discrete control          | Converges slowly but steadily |
| MountainCarContinuous | Continuous actions        | Needs good normalization      |
| LunarLander-v2        | Complex rewards           | Sensitive to learning rate    |
| Pendulum-v1           | Continuous torque control | Requires baseline             |

#### Complexity

- Time: $O(T)$ per episode (trajectory length)
- Space: $O(T)$ (store rewards and gradients)
- Convergence: Unbiased but high-variance; slow without baseline

REINFORCE shows the simplest truth in reinforcement learning, 
increase the probability of what worked.
Every successful action leaves a small gradient trail toward better behavior.

### 968. Actor–Critic (Value-Guided Policy Update)

Actor–Critic algorithms blend two ideas, the actor learns *what to do*, while the critic learns *how good* that decision was.
This fusion of policy gradient and value estimation makes learning faster and more stable than pure policy-based (like REINFORCE) or pure value-based (like Q-Learning) methods.

#### What Problem Are We Solving?

REINFORCE learns directly from entire episodes, which causes:

- High variance (updates depend on long-term rewards)
- Slow learning (no intermediate feedback)

Actor–Critic solves this by adding a critic that estimates the value function $V_\phi(s)$, providing the actor with real-time feedback at every step.
It bridges the gap between Monte Carlo and temporal-difference learning.

#### The Core Idea

We maintain two networks:

- Actor: the policy $\pi_\theta(a|s)$ that decides what to do
- Critic: the value function $V_\phi(s)$ that judges how good the current state (or action) is

The actor updates in the direction suggested by the critic's feedback:
$$
\nabla_\theta J(\theta) = \mathbb{E} \big[ \nabla_\theta \log \pi_\theta(a_t|s_t) , \delta_t \big]
$$
where the TD error (advantage) is:
$$
\delta_t = R_{t+1} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
$$

The critic learns to minimize the mean squared TD error:
$$
L(\phi) = \big( \delta_t \big)^2
$$

#### How It Works (Plain Language)

Think of the actor as an explorer and the critic as a coach.
The actor tries different actions, and the critic immediately says "that was better/worse than expected."
The actor then adjusts its probabilities accordingly, while the critic keeps refining its sense of value.

#### Step-by-Step Summary

| Step | Description                                                                                        |       |
| ---- | -------------------------------------------------------------------------------------------------- | ----- |
| 1    | Initialize actor parameters $\theta$ and critic parameters $\phi$                                  |       |
| 2    | For each episode:                                                                                  |       |
|      | a. Observe state $s_t$, choose action $a_t \sim \pi_\theta(a_t                                     | s_t)$ |
|      | b. Execute $a_t$, observe $R_{t+1}$ and $s_{t+1}$                                                  |       |
|      | c. Compute TD error: $\delta_t = R_{t+1} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$                   |       |
|      | d. Update critic: $\phi \leftarrow \phi + \beta , \delta_t , \nabla_\phi V_\phi(s_t)$              |       |
|      | e. Update actor: $\theta \leftarrow \theta + \alpha , \delta_t , \nabla_\theta \log \pi_\theta(a_t | s_t)$ |
|      | f. Repeat until done                                                                               |       |

#### Mathematical Formulation

1. Actor update
   $$
   \theta \leftarrow \theta + \alpha , \delta_t , \nabla_\theta \log \pi_\theta(a_t|s_t)
   $$

2. Critic update
   $$
   \phi \leftarrow \phi + \beta , \delta_t , \nabla_\phi V_\phi(s_t)
   $$

3. TD error (advantage)
   $$
   \delta_t = R_{t+1} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
   $$

#### Example

Consider a robot learning to walk:

- The actor controls muscle activations (policy)
- The critic predicts future stability (value)
  If a step improves balance, the critic's $\delta_t$ is positive, rewarding those actions.
  If the robot stumbles, $\delta_t$ becomes negative, and the actor reduces those action probabilities.

#### Tiny Code

Python (simplified actor–critic skeleton)

```python
import torch, torch.nn as nn, torch.optim as optim
import numpy as np

class Actor(nn.Module):
    def __init__(self, n_states, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, 128), nn.ReLU(),
            nn.Linear(128, n_actions), nn.Softmax(dim=-1))
    def forward(self, x):
        return self.net(x)

class Critic(nn.Module):
    def __init__(self, n_states):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_states, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, x):
        return self.net(x)

actor, critic = Actor(4, 2), Critic(4)
opt_a = optim.Adam(actor.parameters(), lr=1e-3)
opt_c = optim.Adam(critic.parameters(), lr=1e-3)
gamma = 0.99

def update(s, a, r, s_next):
    s = torch.tensor(s, dtype=torch.float32)
    s_next = torch.tensor(s_next, dtype=torch.float32)
    v_s = critic(s)
    v_next = critic(s_next).detach()
    delta = r + gamma * v_next - v_s
    # critic update
    loss_c = delta.pow(2)
    opt_c.zero_grad(); loss_c.backward(); opt_c.step()
    # actor update
    probs = actor(s)
    log_prob = torch.log(probs[a])
    loss_a = -log_prob * delta.detach()
    opt_a.zero_grad(); loss_a.backward(); opt_a.step()
```

#### Why It Matters

- Lower variance than REINFORCE (thanks to the critic)
- Online learning, can update after each step instead of full episodes
- Foundation for advanced methods like A2C, PPO, DDPG, and SAC
- Balances exploration and exploitation through continuous feedback

#### A Gentle Proof (Why It Works)

The policy gradient theorem with a baseline $b(s)=V(s)$ gives:
$$
\nabla_\theta J(\theta) = \mathbb{E}*{\pi*\theta} \big[ \nabla_\theta \log \pi_\theta(a|s) (Q(s,a) - V(s)) \big]
$$

The critic approximates $V(s)$, so $Q(s,a) - V(s) \approx \delta_t$.
Thus, the actor update uses the TD error $\delta_t$ as an unbiased estimator of advantage.

#### Try It Yourself

1. Implement Actor–Critic for CartPole-v1.
2. Compare performance with REINFORCE, notice smoother learning.
3. Try continuous actions using Gaussian policy.
4. Add entropy regularization to encourage exploration.
5. Experiment with learning rates $\alpha$ and $\beta$.

#### Test Cases

| Environment    | Description        | Observation                         |
| -------------- | ------------------ | ----------------------------------- |
| CartPole-v1    | Discrete control   | Faster convergence than REINFORCE   |
| Pendulum-v1    | Continuous control | Works with Gaussian policies        |
| LunarLander-v2 | Complex dynamics   | Stable learning with small variance |
| MountainCar-v0 | Sparse rewards     | Needs patience and exploration      |

#### Complexity

- Time: $O(T)$ per episode (TD updates each step)
- Space: $O(|\theta| + |\phi|)$
- Convergence: faster and smoother than REINFORCE, but sensitive to critic bias

Actor–Critic is like a two-part mind, 
the actor acts with intuition, the critic evaluates with reason,
and together they converge toward mastery.

### 969. PPO (Proximal Policy Optimization)

Proximal Policy Optimization (PPO) is one of the most popular and stable policy gradient algorithms.
It improves upon Actor–Critic methods by carefully controlling how much the policy can change at each update, preventing destructive steps that destabilize training.

#### What Problem Are We Solving?

In policy gradient and actor–critic methods, large policy updates can cause:

- Instability: new policies deviate too much from previous ones
- Collapse: overfitting to noisy advantages
- Catastrophic forgetting: good behaviors are suddenly lost

PPO solves this by enforcing a *trust region*, it limits how far the new policy can move from the old one while still improving.

#### The Core Idea

PPO optimizes a clipped surrogate objective:
$$
L^{CLIP}(\theta) = \mathbb{E}_t \Big[ \min\big( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \big) \Big]
$$

where:

- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ is the probability ratio
- $A_t$ is the advantage estimate
- $\epsilon$ (e.g., 0.1–0.3) defines the clipping range

The "min" expression ensures that when $r_t(\theta)$ moves too far from 1, the gradient is cut off, keeping updates safe and stable.

#### How It Works (Plain Language)

Think of PPO as telling the actor:

> "You can improve the policy, but only a little at a time."

It prevents the actor from "jumping off a cliff" by limiting how much the probability of chosen actions can change.
This controlled adjustment keeps learning steady even in complex environments.

#### Step-by-Step Summary

| Step | Description                                                                         |
| ---- | ----------------------------------------------------------------------------------- |
| 1    | Collect experience $(s_t, a_t, r_t, s_{t+1})$ using the current policy $\pi_\theta$ |
| 2    | Estimate the advantage $A_t = R_t + \gamma V(s_{t+1}) - V(s_t)$                 |
| 3    | Compute the probability ratio $r_t(\theta)$                                     |
| 4    | Update policy by maximizing the clipped objective $L^{CLIP}(\theta)$            |
| 5    | Update value function $V_\phi(s)$ by minimizing squared error                       |
| 6    | Repeat for several epochs over the same batch (mini-batch updates)                  |

#### Mathematical Formulation

1. Policy loss (clipped objective):
   $$
   L^{CLIP}(\theta) = \mathbb{E}\Big[\min(r_t(\theta)A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t)\Big]
   $$

2. Value loss:
   $$
   L^{V}(\phi) = \mathbb{E}\Big[ (V_\phi(s_t) - R_t)^2 \Big]
   $$

3. Entropy regularization (optional):
   $$
   L^{S} = \mathbb{E}[ -\beta , H(\pi_\theta(\cdot|s_t)) ]
   $$

4. Combined objective:
   $$
   L = L^{CLIP} - c_1 L^{V} + c_2 L^{S}
   $$

#### Example

Suppose the old policy predicted:
$$
\pi_{\text{old}}(a|s) = 0.5, \quad A_t = +2
$$
If the new policy predicts $\pi_\theta(a|s) = 0.8$,
then $r_t = 1.6$, and since it exceeds $(1+\epsilon)$, the gradient is clipped, ensuring stable improvement.

#### Tiny Code

Python (simplified PPO training loop)

```python
import torch, torch.nn as nn, torch.optim as optim

class ActorCritic(nn.Module):
    def __init__(self, n_states, n_actions):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(n_states, 64), nn.Tanh())
        self.actor = nn.Sequential(nn.Linear(64, n_actions), nn.Softmax(dim=-1))
        self.critic = nn.Linear(64, 1)
    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h)

env_n_states, env_n_actions = 4, 2
model = ActorCritic(env_n_states, env_n_actions)
optimizer = optim.Adam(model.parameters(), lr=3e-4)
epsilon = 0.2
gamma = 0.99

def ppo_update(batch):
    s, a, r, s_next, old_logprob, adv = batch
    probs, values = model(torch.tensor(s, dtype=torch.float32))
    dist = torch.distributions.Categorical(probs)
    logprob = dist.log_prob(torch.tensor(a))
    ratio = torch.exp(logprob - torch.tensor(old_logprob))
    clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * torch.tensor(adv)
    loss = -torch.min(ratio * torch.tensor(adv), clipped).mean()
    optimizer.zero_grad(); loss.backward(); optimizer.step()
```

#### Why It Matters

- Stability: avoids catastrophic updates
- Efficiency: multiple epochs of mini-batch optimization
- Simplicity: easy to implement (no complex constraints like TRPO)
- Performance: state-of-the-art in robotics, games, and control

PPO is widely used in:

- OpenAI Gym environments
- Robotics (MuJoCo, Isaac Gym)
- Game agents (Atari, StarCraft, Dota 2)
- Sim-to-real control transfer

#### A Gentle Proof (Why It Works)

The key to PPO's success is its surrogate objective bound.
When the policy update is small ($r_t \approx 1$), PPO behaves like a standard policy gradient.
When updates grow too large, the clipping term caps the objective, limiting step size.
This behaves like a *soft trust region*, ensuring monotonic improvement in expected return.

#### Try It Yourself

1. Implement PPO for CartPole-v1 or LunarLander-v2.
2. Compare clipped vs unclipped updates, observe stability difference.
3. Tune $\epsilon$ between 0.1–0.3 for best results.
4. Plot policy ratio $r_t$ over time to visualize constraint effects.
5. Extend to continuous actions using Gaussian distributions.

#### Test Cases

| Environment        | Description        | Observation                         |
| ------------------ | ------------------ | ----------------------------------- |
| CartPole-v1        | Discrete control   | Fast and stable learning            |
| LunarLander-v2     | Sparse rewards     | Smooth training curve               |
| Hopper-v2 (MuJoCo) | Continuous control | Strong performance                  |
| Humanoid-v2        | High-dimensional   | Scales well with mini-batch updates |

#### Complexity

- Time: $O(E \times B)$ (epochs × batch size)
- Space: $O(B)$ (stored trajectories)
- Convergence: stable, monotonic improvement

PPO is the gold standard of modern policy optimization, 
simple to implement, robust to hyperparameters,
and the reason reinforcement learning scaled beyond the lab.

### 970. DDPG / SAC (Continuous Action Reinforcement Learning)

When environments require continuous actions, like steering a car or controlling a robotic arm, discrete algorithms such as Q-learning or PPO need adaptation.
DDPG (Deep Deterministic Policy Gradient) and SAC (Soft Actor–Critic) are two powerful actor–critic frameworks designed for these continuous domains, balancing precision, stability, and exploration.

#### What Problem Are We Solving?

Most reinforcement learning algorithms assume discrete actions (e.g., move left or right).
But real-world control problems often need continuous control (e.g., throttle = 0.73).
For such cases:

- Q-learning cannot directly apply (it requires $\max_a Q(s,a)$).
- Policy gradients (like REINFORCE) are too noisy.
- PPO struggles with fine-grained precision.

DDPG and SAC solve this by combining value-based learning (for stability) and policy-based learning (for flexibility).

### 1. Deep Deterministic Policy Gradient (DDPG)

DDPG is an off-policy, deterministic actor–critic algorithm.
It uses two networks:

- Actor: outputs a deterministic action $a = \mu_\theta(s)$
- Critic: evaluates it via $Q_\phi(s, a)$

#### Core Update Rules

1. Critic update (TD error):
   $$
   L(\phi) = \big(Q_\phi(s_t, a_t) - y_t\big)^2
   $$
   where
   $$
   y_t = r_t + \gamma Q_{\phi'}(s_{t+1}, \mu_{\theta'}(s_{t+1}))
   $$
   (The primed networks are *target networks* for stability.)

2. Actor update (policy gradient):
   $$
   \nabla_\theta J(\theta) = \mathbb{E}\big[ \nabla_a Q_\phi(s,a) \big|*{a=\mu*\theta(s)} \nabla_\theta \mu_\theta(s) \big]
   $$

#### Key Features

- Deterministic actions $\to$ stable updates
- Target networks $\to$ prevent divergence
- Experience replay $\to$ sample efficiency
- Noise (like Ornstein–Uhlenbeck) $\to$ exploration in continuous space

#### Tiny Code (DDPG skeleton)

```python
import torch, torch.nn as nn, torch.optim as optim

class Actor(nn.Module):
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim, 256), nn.ReLU(),
            nn.Linear(256, a_dim), nn.Tanh())
    def forward(self, s): return self.net(s)

class Critic(nn.Module):
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim + a_dim, 256), nn.ReLU(),
            nn.Linear(256, 1))
    def forward(self, s, a): return self.net(torch.cat([s,a], dim=-1))
```

#### Why It Matters

- Excellent for robotic control, autonomous vehicles, and game physics.
- Handles high-dimensional continuous actions smoothly.
- Foundation for more advanced variants like TD3 and SAC.

### 2. Soft Actor–Critic (SAC)

SAC extends DDPG with a maximum entropy principle, encouraging exploration and robustness.
It learns stochastic policies rather than deterministic ones, favoring both reward and entropy.

#### Objective

Maximize the expected return with entropy:
$$
J(\pi) = \sum_t \mathbb{E}_{(s_t,a_t)\sim\pi}\big[ r(s_t,a_t) + \alpha , \mathcal{H}(\pi(\cdot|s_t)) \big]
$$

Here, $\alpha$ controls the exploration–exploitation tradeoff:

- Large $\alpha$ → more random exploration
- Small $\alpha$ → more focused on reward

#### Policy Update

SAC minimizes the soft policy loss:
$$
L_\pi(\theta) = \mathbb{E}\big[ \alpha \log \pi_\theta(a_t|s_t) - Q_\phi(s_t, a_t) \big]
$$

and the critic loss:
$$
L_Q(\phi) = \big(Q_\phi(s_t,a_t) - (r_t + \gamma V_{\bar{\phi}}(s_{t+1}))\big)^2
$$

with the soft value function:
$$
V_{\bar{\phi}}(s) = \mathbb{E}*{a\sim\pi*\theta}\big[ Q_\phi(s,a) - \alpha \log \pi_\theta(a|s) \big]
$$

#### How It Works (Plain Language)

SAC doesn't just seek *good* actions; it prefers many good options.
By rewarding *uncertainty* (entropy), it avoids getting stuck in narrow, brittle behaviors.
The result: smooth, stable, and exploratory control.

#### Step-by-Step Summary

| Step | Description                                                       |
| ---- | ----------------------------------------------------------------- |
| 1    | Collect experiences $(s,a,r,s')$                                  |
| 2    | Update critic with soft target $r + \gamma (Q - \alpha \log \pi)$ |
| 3    | Update actor to maximize $Q - \alpha \log \pi$                    |
| 4    | Adjust $\alpha$ automatically to maintain target entropy          |
| 5    | Repeat using replay buffer samples                                |

#### Tiny Code (SAC skeleton)

```python
def soft_q_loss(s, a, r, s_next, done, actor, critic, target, alpha):
    with torch.no_grad():
        a_next, logp = actor.sample(s_next)
        q_target = r + (1 - done) * gamma * (target(s_next, a_next) - alpha * logp)
    q = critic(s, a)
    return ((q - q_target)2).mean()
```

#### Why It Matters

- More stable than DDPG (thanks to entropy term)
- Better exploration, avoids premature convergence
- Works well in stochastic, high-dimensional tasks
- Automatic entropy tuning simplifies hyperparameters

SAC often outperforms all other continuous-control methods on benchmarks like HalfCheetah, Walker2D, and Ant.

#### Comparison Table

| Feature           | DDPG                 | SAC                        |
| ----------------- | -------------------- | -------------------------- |
| Policy type       | Deterministic        | Stochastic                 |
| Exploration       | Noise process        | Entropy term               |
| Stability         | Moderate             | Very high                  |
| Sample efficiency | High                 | High                       |
| Entropy tuning    | Manual               | Automatic                  |
| Typical domains   | Robotics, simulators | Robotics, physics, control |

#### Try It Yourself

1. Train both DDPG and SAC on Pendulum-v1.
2. Compare learning curves, SAC will be smoother.
3. Adjust $\alpha$ to see entropy's effect on exploration.
4. Extend to Humanoid-v2 for large-scale control.
5. Experiment with replay buffer size and target update rate.

#### Test Cases

| Environment | Algorithm | Observation                   |
| ----------- | --------- | ----------------------------- |
| Pendulum-v1 | DDPG      | Works, but sensitive to noise |
| Hopper-v2   | SAC       | Stable, continuous learning   |
| Ant-v2      | SAC       | Smooth convergence            |
| Reacher-v2  | DDPG      | May overfit or oscillate      |

#### Complexity

- Time: $O(N)$ per batch (two networks updated per step)
- Space: $O(|\theta| + |\phi|)$ (actor and critic params)
- Convergence: SAC > DDPG in stability and exploration

In short:
DDPG aims for precision; SAC aims for stability.
Together, they define the modern standard for continuous control learning, 
where the agent doesn't just choose actions, but *flows* through them.

# Section 98. Approximation and Online Algorithms 

### 971. Greedy Set Cover (ln n-Approximation)

The Set Cover Problem asks: given a universe of elements and a collection of sets that cover them, what is the smallest number of sets needed to cover everything?
This problem is NP-hard, but a simple greedy algorithm gives a logarithmic-factor approximation, one of the most elegant results in approximation algorithms.

#### What Problem Are We Solving?

Formally:
Let the universe be
$$
U = {e_1, e_2, \dots, e_n}
$$
and a family of subsets
$$
S = {S_1, S_2, \dots, S_m}, \quad S_i \subseteq U.
$$

We want to choose the smallest subcollection $C \subseteq S$ such that
$$
\bigcup_{S_i \in C} S_i = U.
$$

This appears in:

- Sensor placement (cover all regions)
- Feature selection
- Test suite minimization
- Network monitoring

Exact solutions are exponential in $m$, so we seek a good approximation instead.

#### The Core Idea

Pick the set that covers the largest number of uncovered elements at each step.
Repeat until every element is covered.

Although greedy may not find the perfect solution, it guarantees a solution no worse than a factor of $\ln n$ times the optimal.

#### Step-by-Step Example

Suppose
$U = {1,2,3,4,5,6}$
and
$$
S_1 = {1,2,3}, \quad S_2 = {2,4}, \quad S_3 = {3,4,5}, \quad S_4 = {5,6}.
$$

Greedy steps:

| Iteration | Sets Available | Chosen Set | Newly Covered | Covered So Far |
| --------- | -------------- | ---------- | ------------- | -------------- |
| 1         | all            | $S_1$      | {1,2,3}       | {1,2,3}        |
| 2         | remaining      | $S_3$      | {4,5}         | {1,2,3,4,5}    |
| 3         | remaining      | $S_4$      | {6}           | {1,2,3,4,5,6}  |

Total sets used = 3 (optimal is 2 or 3).
Still within the $\ln n$ bound.

#### Algorithm (Pseudocode)

```
GreedySetCover(U, S):
    C = ∅
    while U not empty:
        pick S_i in S that covers the largest number of uncovered elements
        C = C ∪ {S_i}
        U = U \ S_i
    return C
```

#### Mathematical Guarantee

Let OPT be the optimal number of sets.
Then the greedy algorithm selects at most
$$
H_n = 1 + \frac{1}{2} + \frac{1}{3} + \dots + \frac{1}{n} \le \ln n + 1
$$
times OPT.

So:
$$
|C_{\text{greedy}}| \le (\ln n + 1) , |C_{\text{opt}}|.
$$

This is the best possible unless $P = NP$.

#### Tiny Code

Python

```python
def greedy_set_cover(universe, sets):
    U = set(universe)
    cover = []
    while U:
        best = max(sets, key=lambda s: len(U & s))
        cover.append(best)
        U -= best
    return cover

U = {1,2,3,4,5,6}
S = [{1,2,3}, {2,4}, {3,4,5}, {5,6}]
C = greedy_set_cover(U, S)
print("Cover:", C)
```

#### Why It Matters

- Greedy set cover is a template for many covering and selection problems.
- It achieves the tight logarithmic approximation bound.
- It's easy to implement and works well in practice.

Applications include:

- Selecting minimal training examples
- Reducing redundancy in feature sets
- Building minimal monitoring systems

#### Try It Yourself

1. Generate random subsets and test how many sets greedy picks vs. optimal (for small $n$).
2. Visualize coverage after each step.
3. Implement the weighted version, prefer sets with higher "value per cost" ratio.
4. Test on document summarization: sentences as sets, words as elements.

#### Test Cases

| Universe               | Sets                      | Result                            |
| ---------------------- | ------------------------- | --------------------------------- |
| {1,2,3,4}              | {{1,2}, {2,3}, {3,4}}     | Picks 3 sets                      |
| {1,2,3,4,5,6}          | {{1,2,3}, {3,4,5}, {5,6}} | Picks 3 sets                      |
| Random subsets (n=100) |,                         | Achieves near-optimal ln(n) ratio |

#### Complexity

- Time: $O(nm)$ (each iteration scans all sets)
- Space: $O(n+m)$
- Approximation ratio: $\ln n + 1$

The greedy set cover is a small miracle of simplicity, 
a one-line heuristic that quietly reaches the best provable bound for one of computer science's hardest problems.

### 972. Vertex Cover Approximation (Double-Matching Heuristic)

The Vertex Cover Problem asks for the smallest set of vertices that touch every edge in a graph.
It's a cornerstone NP-hard problem, but surprisingly, a very simple algorithm achieves a 2-approximation, meaning it's at most twice as large as the optimal solution.

#### What Problem Are We Solving?

Given an undirected graph
$$
G = (V, E)
$$
find the smallest subset of vertices
$$
C \subseteq V
$$
such that for every edge $(u, v) \in E$, at least one of $u$ or $v$ is in $C$.

In other words, all edges are "covered" by selecting enough vertices.

Applications include:

- Network monitoring (every link has a watcher)
- Facility placement (each connection served by one site)
- Resource allocation and security systems

#### The Core Idea

Pick any uncovered edge $(u,v)$, add both its endpoints to the cover,
then remove all edges incident to either $u$ or $v$.
Repeat until no edges remain.

This simple greedy-like approach guarantees that:
$$
|C_{\text{approx}}| \le 2 , |C_{\text{opt}}|
$$

#### Step-by-Step Example

Graph:

```
Edges = {(1,2), (2,3), (3,4), (4,5)}
```

Algorithm:

| Step | Picked Edge | Added Vertices | Remaining Edges       | Current Cover |
| ---- | ----------- | -------------- | --------------------- | ------------- |
| 1    | (1,2)       | {1,2}          | (3,4), (4,5)          | {1,2}         |
| 2    | (3,4)       | {3,4}          | (4,5) already covered | {1,2,3,4}     |

Final cover: {1,2,3,4}.
Optimal cover is {2,4} (size 2),
so ratio = 4 / 2 = 2, perfectly within the bound.

#### Algorithm (Pseudocode)

```
ApproximateVertexCover(G):
    C = ∅
    while E not empty:
        pick any edge (u, v) ∈ E
        C = C ∪ {u, v}
        remove all edges incident to u or v
    return C
```

#### Mathematical Guarantee

Let OPT be the minimum vertex cover.
The algorithm picks both endpoints for each selected edge,
but since any edge must be incident to at least one vertex in OPT,
the number of chosen edges ≤ |OPT|,
and hence:

$$
|C| \le 2 |OPT|
$$

So the algorithm is a 2-approximation.

#### Tiny Code

Python

```python
def vertex_cover_approx(graph):
    cover = set()
    edges = set(graph)
    while edges:
        u, v = edges.pop()
        cover.update([u, v])
        edges = {e for e in edges if u not in e and v not in e}
    return cover

edges = {(1,2), (2,3), (3,4), (4,5)}
print(vertex_cover_approx(edges))
```

#### Why It Matters

- Fast, simple, and provably near-optimal
- Basis for numerous network optimization heuristics
- Helps design better bounds for other graph problems (like set cover, clique cover, etc.)
- Forms part of "primal–dual" frameworks in combinatorial optimization

#### Try It Yourself

1. Generate random graphs and compare to optimal vertex cover via brute force (for $n < 10$).
2. Implement weighted vertex cover (prefer lower-cost vertices).
3. Visualize the graph, mark edges as "covered" after each iteration.
4. Compare with matching-based approximation (maximum matching gives same factor).

#### Test Cases

| Graph               | Approx Cover | Optimal | Ratio |
| ------------------- | ------------ | ------- | ----- |
| Path (1–2–3–4–5)    | {1,2,3,4}    | {2,4}   | 2.0   |
| Triangle (1–2–3–1)  | {1,2}        | {1,2}   | 1.0   |
| Star (1–{2,3,4,5})  | {1,2}        | {1}     | 2.0   |
| Random 6-node graph | ~2×          | ~1×     | ≤ 2.0 |

#### Complexity

- Time: $O(|E|)$
- Space: $O(|V|)$
- Approximation ratio: ≤ 2

#### A Gentle Proof (Why It Works)

Each chosen edge $(u,v)$ contributes two vertices to the cover.
In any optimal cover, at least one of $u$ or $v$ must appear, hence each edge contributes at least one "token" toward OPT.
Since we may pick both, we overcount by at most a factor of 2.

#### Summary Table

| Property       | Value                            |
| -------------- | -------------------------------- |
| Algorithm Type | Greedy / Matching-based          |
| Guarantee      | 2× optimal                       |
| Determinism    | Yes                              |
| Works On       | Unweighted graphs                |
| Extension      | Weighted version via LP rounding |

The vertex cover approximation is a perfect example of beauty in simplicity, 
one small doubling step that brings a hard combinatorial puzzle within reach of practicality.

### 973. Traveling Salesman Approximation (MST-based 2-Approximation)

The Traveling Salesman Problem (TSP) asks for the shortest possible tour that visits all cities exactly once and returns to the start.
It's one of the most famous NP-hard problems in combinatorial optimization, yet, when distances satisfy the triangle inequality, there's a simple and elegant 2-approximation algorithm based on the Minimum Spanning Tree (MST).

#### What Problem Are We Solving?

Given a complete weighted graph
$$
G = (V, E)
$$
where $w(u,v)$ is the distance (cost) between vertices $u$ and $v$,
find a cycle that visits all vertices exactly once with minimal total cost.

Formally:

$$
\min_{\pi} \sum_{i=1}^{n} w(\pi_i, \pi_{i+1})
$$
where $\pi$ is a permutation of the vertices and $\pi_{n+1} = \pi_1$.

When the distances satisfy the triangle inequality:

$$
w(u,v) \le w(u,x) + w(x,v),
$$

we can approximate the optimal tour using MST doubling and traversal.

#### The Core Idea

1. Compute an MST of the graph (minimum cost to connect all nodes).
2. Traverse the MST in preorder (depth-first traversal).
3. Shortcut repeated vertices using triangle inequality.

The resulting tour's total cost ≤ 2 × optimal.

#### Step-by-Step Example

Consider a complete graph of 4 cities:

| Edge  | Cost |
| ----- | ---- |
| (A,B) | 1    |
| (A,C) | 3    |
| (A,D) | 4    |
| (B,C) | 2    |
| (B,D) | 5    |
| (C,D) | 3    |

Step 1: Build MST
Edges in MST = {(A,B), (B,C), (C,D)}
Total MST cost = 1 + 2 + 3 = 6.

Step 2: Traverse preorder: A → B → C → D → A
Raw traversal cost = 8.

Step 3: Shortcut repeats if necessary (none here).
Final TSP tour = 8 ≤ 2 × 6 = 12 (within bound).

#### Algorithm (Pseudocode)

```
ApproxTSPviaMST(G):
    T = MinimumSpanningTree(G)
    tour = PreorderTraversal(T)
    return tour with shortcuts
```

#### Mathematical Guarantee

Let OPT be the cost of the optimal TSP tour.

- The MST has cost ≤ OPT (since removing one edge from the optimal tour yields a spanning tree).
- Traversing the MST twice gives total cost ≤ 2 × MST ≤ 2 × OPT.
- Shortcutting (triangle inequality) never increases cost.

Thus:

$$
\text{Cost}*{\text{approx}} \le 2 \times \text{Cost}*{\text{OPT}}.
$$

#### Tiny Code

Python

```python
import networkx as nx

def tsp_mst_approx(G):
    T = nx.minimum_spanning_tree(G)
    preorder = list(nx.dfs_preorder_nodes(T, source=list(T.nodes)[0]))
    tour = preorder + [preorder[0]]  # complete the cycle
    cost = sum(G[tour[i]][tour[i+1]]['weight'] for i in range(len(tour)-1))
    return tour, cost
```

#### Why It Matters

- Simple yet effective for metric TSPs (triangle inequality holds).
- Foundation for better heuristics (Christofides, 1.5-approximation).
- Applies in routing, network cabling, robot motion, delivery logistics, etc.

The MST heuristic captures a "skeleton" of efficient connectivity, 
doubling it ensures coverage, and shortcutting cleans up redundancy.

#### Try It Yourself

1. Generate random 2D points and compute Euclidean distances.
2. Compare MST-based TSP cost to brute-force optimal (for $n \le 10$).
3. Visualize MST vs. resulting TSP path.
4. Extend to Christofides Algorithm (add minimum matching on odd-degree vertices).

#### Test Cases

| Graph Type       | Approximation | Observation          |
| ---------------- | ------------- | -------------------- |
| 4-city Euclidean | ≤ 2×          | Matches bound        |
| 10 random cities | ≤ 2×          | Works consistently   |
| Metric graph     | ≤ 2×          | Guaranteed bound     |
| Non-metric graph | > 2×          | Bound not guaranteed |

#### Complexity

- Time: $O(E \log V)$ for MST + $O(V)$ for traversal
- Space: $O(V)$
- Approximation ratio: ≤ 2

#### A Gentle Proof (Why It Works)

Let $T$ be the MST, and let $H$ be the Euler tour obtained by doubling each edge in $T$.
Since $T$ is connected, $H$ visits every vertex at least once and has cost $2w(T)$.
By shortcutting repeated vertices (using triangle inequality), we get a Hamiltonian cycle whose cost ≤ $2w(T)$.
As $w(T) \le w(\text{OPT})$, we conclude:

$$
w(\text{tour}) \le 2 w(\text{OPT}).
$$

#### Summary Table

| Property            | Value                   |
| ------------------- | ----------------------- |
| Algorithm Type      | MST-based heuristic     |
| Approximation Ratio | ≤ 2                     |
| Requires            | Triangle inequality     |
| Determinism         | Yes                     |
| Extension           | Christofides (1.5× OPT) |

The MST-based TSP approximation is the perfect balance between mathematical beauty and computational practicality, 
it doubles what you need, then prunes what you don't.

### 974. k-Center Approximation (Farthest-Point Heuristic)

The k-Center Problem asks: given a set of points and a distance metric, how can we pick *k* centers so that the maximum distance from any point to its nearest center is as small as possible?
It's a classic clustering and facility-location problem, NP-hard, but solvable within a 2-approximation using a simple farthest-point heuristic.

#### What Problem Are We Solving?

Given a set of points $V$ and a distance function $d(u,v)$,
select $k$ centers $C = {c_1, c_2, \dots, c_k}$
to minimize the maximum distance from any point to its nearest center:

$$
r^* = \min_{C \subseteq V, |C|=k} \max_{v \in V} \min_{c \in C} d(v, c)
$$

The goal is to make sure no point is too far from a chosen center.

This models problems like:

- Placing hospitals to minimize the farthest patient distance
- Designing data centers for minimal latency
- Building cell towers with guaranteed coverage radius

#### The Core Idea

1. Start with an arbitrary point as the first center.
2. Repeatedly choose the farthest point from all current centers.
3. Stop after $k$ centers have been chosen.

This greedy selection ensures every new center captures the worst-served region so far.
The result is at most twice as far as the optimal radius.

#### Step-by-Step Example

Points along a line:
$V = {0, 1, 2, 5, 8, 11}$ and $k = 2$.

Algorithm:

| Step | Chosen Centers       | Farthest Point | Distance to Closest Center | Centers After Step |
| ---- | -------------------- | -------------- | -------------------------- | ------------------ |
| 1    | pick arbitrary → {0} | farthest = 11  | distance = 11              | {0, 11}            |
| 2    | stop (2 centers)     |,              |,                          | final = {0, 11}    |

Radius achieved:
Every point ≤ 5.5 from nearest center.
Optimal radius: 5 → so ratio = 1.1 ≤ 2.

#### Algorithm (Pseudocode)

```
GreedyKCenter(V, k):
    pick any point v as first center
    C = {v}
    while |C| < k:
        select point x in V that maximizes min_{c in C} d(x, c)
        add x to C
    return C
```

#### Mathematical Guarantee

Let $r^*$ be the optimal radius.
At every step, the algorithm picks a point at least $r^*$ away from existing centers,
ensuring that the chosen centers are all at least $r^*$ apart.
When $k$ centers are selected, every remaining point is within $2r^*$.

So:

$$
r_{\text{greedy}} \le 2r^*
$$

#### Tiny Code

Python

```python
import numpy as np

def k_center(points, k):
    centers = [points[0]]
    while len(centers) < k:
        dist = [min(np.linalg.norm(p - c) for c in centers) for p in points]
        next_center = points[np.argmax(dist)]
        centers.append(next_center)
    return np.array(centers)
```

#### Why It Matters

- One of the simplest 2-approximation algorithms in combinatorial optimization
- Scales linearly with input size, ideal for clustering large datasets
- Intuitive and geometric, repeatedly covers the most distant area
- Foundation for facility location, graph partitioning, and distributed clustering

#### Try It Yourself

1. Generate random 2D points and run the algorithm for different $k$.
2. Plot the centers, they should spread out uniformly.
3. Compare to K-means (minimizes average distance, not max distance).
4. Observe how adding one more center dramatically reduces the maximum radius.
5. Test on graph distances instead of Euclidean distances.

#### Test Cases

| Dataset       | k | Approx Radius | Optimal Radius          | Ratio |
| ------------- | - | ------------- | ----------------------- | ----- |
| Line [0–10]   | 2 | 5.5           | 5.0                     | 1.1   |
| Grid 3×3      | 3 | 1.9           | 1.0                     | ≤ 2   |
| Random 50 pts | 5 | ~2× optimal   | consistent              |       |
| City graph    | 4 | ≤ 2×          | holds for metric graphs |       |

#### Complexity

- Time: $O(k |V|^2)$ (can be optimized using distance caching)
- Space: $O(|V|)$
- Approximation ratio: ≤ 2

#### A Gentle Proof (Why It Works)

Let $C^*$ be the optimal centers with radius $r^*$.
When the greedy algorithm chooses centers, every new center must lie in a distinct ball of radius $r^*$ around an optimal center.
Thus, after $k$ choices, all optimal clusters are represented.
Every remaining point is within at most two radii of a chosen center.

$$
r_{\text{greedy}} \le 2r^*
$$

#### Summary Table

| Property            | Value                                    |
| ------------------- | ---------------------------------------- |
| Algorithm Type      | Greedy (farthest-point)                  |
| Objective           | Minimize max distance                    |
| Approximation Ratio | ≤ 2                                      |
| Metric Required     | Yes (triangle inequality)                |
| Use Cases           | Coverage, clustering, facility placement |

The farthest-point heuristic captures the essence of coverage problems, 
expand evenly, reach farthest first, and guarantee every point has a home within twice the best possible distance.

### 975. Online Paging (LRU – Least Recently Used)

The Online Paging Problem models how an operating system or cache manages a limited amount of fast memory (cache) while serving a sequence of requests.
Since future requests are unknown, the algorithm must make decisions on the fly about which item to evict, a core example of an online algorithm.

#### What Problem Are We Solving?

We have:

- A cache that holds up to $k$ pages.
- A sequence of page requests $p_1, p_2, \dots, p_n$.
- Each request costs 1 if the page is not in cache (a *miss*), and 0 if it is (a *hit*).

When the cache is full and a new page is requested, we must evict one to make space, but we don't know future requests.

Goal:
Minimize the total number of cache misses.

#### The Core Idea (LRU Strategy)

Least Recently Used (LRU) evicts the page that hasn't been accessed for the longest time.
It relies on the temporal locality assumption: pages used recently are likely to be used again soon.

LRU keeps a record of access order and always removes the oldest entry.

#### Example

Cache size $k = 3$
Request sequence: `A, B, C, A, D, B, A, E`

| Step | Request | Cache before | Action           | Cache after | Miss/Hit |
| ---- | ------- | ------------ | ---------------- | ----------- | -------- |
| 1    | A       | {}           | Load A           | {A}         | Miss     |
| 2    | B       | {A}          | Load B           | {A,B}       | Miss     |
| 3    | C       | {A,B}        | Load C           | {A,B,C}     | Miss     |
| 4    | A       | {A,B,C}      | Hit              | {A,B,C}     | Hit      |
| 5    | D       | {A,B,C}      | Evict B (oldest) | {A,C,D}     | Miss     |
| 6    | B       | {A,C,D}      | Evict C          | {A,D,B}     | Miss     |
| 7    | A       | {A,D,B}      | Hit              | {A,D,B}     | Hit      |
| 8    | E       | {A,D,B}      | Evict D          | {A,B,E}     | Miss     |

Total: 8 requests, 6 misses, 2 hits.

#### Algorithm (Pseudocode)

```
LRU(CacheSize, Requests):
    Cache = empty list
    for page in Requests:
        if page in Cache:
            move page to front (most recent)
        else:
            if len(Cache) == CacheSize:
                remove last element (least recent)
            insert page at front
```

#### Mathematical Guarantee

The competitive ratio of an online algorithm compares its performance to the optimal offline algorithm (which knows the future).

For LRU:

$$
\text{Competitive ratio} = k
$$

Meaning:
LRU's total cost is at most $k$ times the cost of the optimal offline algorithm (denoted OPT).

#### Tiny Code

Python

```python
from collections import deque

def lru(cache_size, requests):
    cache = deque()
    hits = 0
    for page in requests:
        if page in cache:
            cache.remove(page)
            cache.appendleft(page)
            hits += 1
        else:
            if len(cache) == cache_size:
                cache.pop()
            cache.appendleft(page)
    return hits, len(requests) - hits

reqs = ['A','B','C','A','D','B','A','E']
print(lru(3, reqs))
```

#### Why It Matters

- LRU is used in CPU caches, OS memory management, web caching, and databases.
- Demonstrates online algorithm design, decision-making without future knowledge.
- Key for studying competitive analysis and worst-case guarantees.

#### Try It Yourself

1. Compare LRU, FIFO, and Random Replacement.
2. Simulate sequences with repeated vs. random access patterns.
3. Increase cache size $k$, observe diminishing misses.
4. Measure competitive ratio experimentally.

#### Test Cases

| Sequence    | Cache Size | Algorithm | Misses  | Hits |
| ----------- | ---------- | --------- | ------- | ---- |
| A,B,C,A,B,C | 2          | LRU       | 4       | 2    |
| A,B,C,A,D,B | 3          | LRU       | 5       | 1    |
| Random      | 4          | LRU       | ≈ k×OPT |,    |

#### Complexity

- Time: $O(nk)$ naive, or $O(n)$ with hash map + linked list
- Space: $O(k)$
- Competitive ratio: ≤ $k$

#### A Gentle Proof (Why It Works)

In any sequence, consider the last $k$ distinct pages accessed before a miss, those must all be in the optimal cache as well.
Since LRU only evicts the least recently used page, it can be at most $k$ times worse than OPT.

Formally:
$$
\text{Cost(LRU)} \le k \cdot \text{Cost(OPT)}
$$

#### Summary Table

| Property          | Value                    |
| ----------------- | ------------------------ |
| Type              | Online, deterministic    |
| Competitive Ratio | ≤ k                      |
| Cache Policy      | Least Recently Used      |
| Works Well On     | Locality-based sequences |
| Used In           | OS, CPU cache, databases |

LRU is the timeless balance between intuition and theory, 
an algorithm that forgets just enough of the past to keep up with an unpredictable future.

### 976. Online Matching (Ranking Algorithm)

The Online Bipartite Matching Problem models situations where one side of a bipartite graph (say, users) arrives one by one, and we must decide immediately which resource (say, server or ad slot) to match them with, without knowing future arrivals.
The Ranking Algorithm achieves a competitive ratio of $(1 - 1/e)$, the best possible for randomized algorithms in this setting.

#### What Problem Are We Solving?

We are given a bipartite graph
$$
G = (U, V, E)
$$
where $U$ (offline vertices) are known in advance, and $V$ (online vertices) arrive one by one.

When each vertex $v \in V$ arrives:

- We see its incident edges $(u,v)$ for all $u \in U$.
- We must immediately and irrevocably choose whether to match $v$ with an unmatched $u$.

Goal:
Maximize the total number of matches made by the end.

#### The Core Idea (Ranking Algorithm)

The Ranking Algorithm is a randomized strategy introduced by Karp, Vazirani, and Vazirani (1990).
It assigns a random ranking to the offline vertices once at the start and uses it to break ties consistently.

Algorithm outline:

1. Randomly permute the offline vertices $U$.
2. For each arriving vertex $v \in V$:

   * Among all currently available neighbors, match $v$ to the highest-ranked (lowest-numbered) one.

This simple strategy achieves a $(1 - 1/e) ≈ 0.632$ competitive ratio, provably optimal for this problem.

#### Example

Let $U = {A, B, C}$ and $V = {1, 2, 3}$ arrive in order.

Edges:

- 1 connects to {A, B}
- 2 connects to {B, C}
- 3 connects to {A, C}

Random ranking of $U$: A=1, B=2, C=3

Step-by-step:

| Arrival | Neighbors | Free?          | Picked Match     |
| ------- | --------- | -------------- | ---------------- |
| 1       | {A, B}    | all free       | A (highest rank) |
| 2       | {B, C}    | all free       | B                |
| 3       | {A, C}    | A used, C free | C                |

Final matching = {(1,A), (2,B), (3,C)} (perfect match).

#### Algorithm (Pseudocode)

```
RankingMatching(U, V, E):
    assign random rank π(u) to each u in U
    for each arriving v in V:
        N = {u ∈ U | (u,v) ∈ E and u unmatched}
        if N ≠ ∅:
            pick u in N with smallest π(u)
            match (u,v)
```

#### Mathematical Guarantee

Let OPT denote the number of matches the optimal offline algorithm can make (knowing the full sequence).

Then:

$$
E[\text{matches(Ranking)}] \ge (1 - 1/e) \cdot \text{OPT}
$$

This bound is tight for all randomized online algorithms under adversarial arrival.

#### Tiny Code

Python

```python
import random

def online_matching_ranking(U, edges, arrivals):
    rank = {u: i for i, u in enumerate(random.sample(U, len(U)))}
    matched = {}
    for v in arrivals:
        candidates = [u for u in U if (u, v) in edges and u not in matched.values()]
        if candidates:
            chosen = min(candidates, key=lambda u: rank[u])
            matched[v] = chosen
    return matched
```

#### Why It Matters

- Fundamental in online advertising, task assignment, resource allocation.
- Balances randomization and greediness to achieve provable guarantees.
- Foundational model for competitive analysis and online learning algorithms.

#### Try It Yourself

1. Construct a bipartite graph where multiple matches are possible.
2. Simulate arrivals in different orders.
3. Compare Ranking vs. Greedy (match to any available).
4. Measure average matches over random rankings.
5. Observe that Ranking consistently beats Greedy in adversarial sequences.

#### Test Cases

| Scenario            | Algorithm | Matches    | Competitive Ratio |
| ------------------- | --------- | ---------- | ----------------- |
| Simple 3×3 complete | Ranking   | 3          | 1.0               |
| Adversarial order   | Ranking   | 0.63×OPT   | ≈ (1 - 1/e)       |
| Random graph        | Ranking   | ≥ 0.63×OPT | consistent        |

#### Complexity

- Time: $O(|E|)$
- Space: $O(|U| + |V|)$
- Competitive Ratio: $(1 - 1/e)$

#### A Gentle Proof (Why It Works)

Random ranking gives each offline vertex an equal chance to be available when its best possible online partner arrives.
Using probabilistic analysis, the expected fraction of matched vertices satisfies:

$$
\frac{dM}{dx} = 1 - M, \quad \Rightarrow \quad M = 1 - e^{-x}
$$

At full capacity ($x=1$), $M = 1 - 1/e$.
Thus, expected performance is $(1 - 1/e)$ times optimal.

#### Summary Table

| Property              | Value                           |
| --------------------- | ------------------------------- |
| Type                  | Online randomized               |
| Competitive Ratio     | $(1 - 1/e)$                     |
| Deterministic Version | Greedy (worse: 0.5×OPT)         |
| Used In               | Online ads, resource allocation |
| Key Idea              | Pre-rank offline side once      |

The Ranking Algorithm is a gem of online optimization, 
simple randomness at the start yields deep robustness against any future.

### 977. Online Knapsack (Ratio-Based Acceptance)

The Online Knapsack Problem models real-time decision-making under capacity constraints:
items arrive one by one, each with a value and weight, and you must decide immediately whether to accept it, without knowing what comes next.
This problem captures ad auctions, job scheduling, and real-time resource allocation under uncertainty.

#### What Problem Are We Solving?

We are given:

- A knapsack of capacity $W$
- A sequence of items $(v_i, w_i)$ arriving one at a time
  where $v_i$ is the value, and $w_i$ is the weight.

For each item:

- If accepted, it consumes $w_i$ capacity.
- The decision is irrevocable, we cannot remove items later.

Goal:
Maximize the total value of accepted items while keeping total weight $\le W$.

This is the online version of the classical 0/1 knapsack problem.

#### The Core Idea (Greedy Ratio Threshold)

Without future knowledge, we can't plan perfectly.
Instead, we use a threshold strategy based on the value-to-weight ratio:

$$
\rho_i = \frac{v_i}{w_i}
$$

Only accept items whose ratio $\rho_i$ is above a dynamic threshold, which is gradually lowered as the knapsack fills.

#### Algorithm Intuition

Think of the knapsack as being filled gradually.
At first, we are picky, we only take high-value items.
As capacity decreases, we lower the threshold and accept lower-value ones.

If we define $x$ as the fraction of capacity filled,
then a common threshold rule is:

$$
\rho(x) = \rho_{\max} e^{x-1}
$$

An arriving item with $\rho_i \ge \rho(x)$ is accepted.

This ensures we use capacity smoothly while retaining near-optimal value.

#### Example

Capacity $W = 10$

| Item | Value ($v_i$) | Weight ($w_i$) | Ratio ($v_i/w_i$) | Decision                   |
| ---- | ------------- | -------------- | ----------------- | -------------------------- |
| 1    | 20            | 4              | 5.0               | accept                     |
| 2    | 10            | 4              | 2.5               | reject (too low)           |
| 3    | 15            | 3              | 5.0               | accept                     |
| 4    | 8             | 4              | 2.0               | reject                     |
| 5    | 12            | 3              | 4.0               | accept (enough space left) |

Total value = 20 + 15 + 12 = 47
Total weight = 4 + 3 + 3 = 10

#### Algorithm (Pseudocode)

```
OnlineKnapsack(W, items):
    used = 0
    accepted = []
    for (v, w) in items:
        rho = v / w
        x = used / W
        threshold = rho_max * exp(x - 1)
        if used + w <= W and rho >= threshold:
            accepted.append((v, w))
            used += w
    return accepted
```

#### Competitive Ratio

Let OPT be the value obtained by the optimal offline algorithm.

For continuous arrivals and differentiable threshold function,
the exponential threshold rule achieves a competitive ratio of $(1 - 1/e)$,
the same as the online matching problem.

Formally:

$$
\frac{E[\text{ALG}]}{\text{OPT}} \ge 1 - \frac{1}{e} \approx 0.632
$$

#### Tiny Code

Python

```python
import math

def online_knapsack(items, W, rho_max):
    used, value = 0, 0
    for v, w in items:
        rho = v / w
        x = used / W
        threshold = rho_max * math.exp(x - 1)
        if used + w <= W and rho >= threshold:
            used += w
            value += v
    return value
```

#### Why It Matters

- Real-world systems (ads, cloud jobs, CPU time) must make allocation decisions immediately.
- Demonstrates the power of exponential thresholds in online optimization.
- Connects deeply to competitive analysis, prophet inequalities, and mechanism design.

#### Try It Yourself

1. Simulate 100 random items with weights and values.
2. Compare your online rule to the offline optimum (sorted by ratio).
3. Adjust the decay function $\rho(x)$ to see its impact.
4. Observe that exponential decay achieves consistently near-optimal results.

#### Test Cases

| Items           | W   | Algorithm             | Competitive Ratio |
| --------------- | --- | --------------------- | ----------------- |
| Random uniform  | 10  | Exponential threshold | ≈ 0.63×OPT        |
| Ad stream       | 100 | Ratio-based           | 0.6–0.65          |
| Constant values | 20  | Any policy            | ≈ OPT             |

#### Complexity

- Time: $O(n)$ (single pass)
- Space: $O(1)$ (only store current fill ratio)
- Competitive Ratio: $(1 - 1/e)$

#### A Gentle Proof (Why It Works)

Let $f(x)$ denote the total expected value at fill fraction $x$.
Differentiating with respect to $x$ under the exponential threshold rule:

$$
\frac{df}{dx} = \rho_{\max} e^{x-1}
$$

Integrating over $x \in [0,1]$ gives:

$$
f(1) = \rho_{\max} (1 - 1/e)
$$

Thus, the total expected value reaches $(1 - 1/e)$ of the best possible.

#### Summary Table

| Property          | Value                                    |
| ----------------- | ---------------------------------------- |
| Type              | Online, ratio-based                      |
| Competitive Ratio | $(1 - 1/e)$                              |
| Decision Rule     | Threshold on $v_i/w_i$                   |
| Used In           | Ad auctions, scheduling, dynamic pricing |
| Optimality        | Tight for adversarial input              |

The Online Knapsack problem embodies the art of tradeoff, 
balancing greed and patience, maximizing value before the future unfolds.

### 978. Competitive Ratio Evaluation

The Competitive Ratio is the central tool used to measure the performance of online algorithms, algorithms that must make decisions *without knowing the future*.
It provides a formal way to compare an online algorithm's result to that of the optimal offline algorithm (which knows the entire input sequence in advance).

#### What Problem Are We Solving?

In online problems, data arrives sequentially, and the algorithm must react instantly.

Examples include:

- Online paging (LRU): evicting pages without future access info
- Online matching (Ranking): assigning ads to users in real time
- Online knapsack: deciding whether to accept new items
- Online scheduling: allocating machines to jobs as they arrive

Since no future knowledge exists, we cannot talk about "optimality" in the traditional sense.
Instead, we measure *how close* the online algorithm can get to the offline optimum.

#### Definition

Let:

- $\text{ALG}(I)$ = cost (or profit) of the online algorithm on input $I$
- $\text{OPT}(I)$ = cost (or profit) of the optimal offline algorithm (knows the future)

Then the competitive ratio $c$ is defined as:

$$
c = \sup_I \frac{\text{ALG}(I)}{\text{OPT}(I)}
$$

for minimization problems, or equivalently

$$
c = \inf_I \frac{\text{ALG}(I)}{\text{OPT}(I)}
$$

for maximization problems.

#### Intuitive Meaning

- If $c = 1$, the online algorithm is *as good as* the offline one, perfectly optimal.
- If $c = 2$, it performs at worst twice as bad.
- If $c = (1 - 1/e)$, it achieves about 63% of the offline optimum, common in randomized algorithms like online matching or knapsack.

Thus, the closer $c$ is to 1, the better the algorithm.

#### Example 1 – Paging (LRU)

For a cache of size $k$:
$$
\text{Cost(LRU)} \le k \cdot \text{Cost(OPT)}
$$

So the competitive ratio is $k$.

LRU can perform $k$ times worse than the optimal offline cache, but no deterministic algorithm can do better.

#### Example 2 – Online Matching (Ranking)

$$
E[\text{ALG}] \ge (1 - 1/e) \cdot \text{OPT}
$$

So the competitive ratio is $(1 - 1/e) \approx 0.632$.

This is provably optimal for randomized online matching algorithms.

#### Example 3 – Ski Rental Problem

You can rent skis each day for cost $r$, or buy them once for cost $b$.

Optimal strategy:

- Rent until total rental cost equals $b$, then buy.

Competitive ratio:

$$
c = \frac{2b - r}{b} \approx 2 - \frac{r}{b}
$$

This gives a 2-competitive deterministic bound.

#### Algorithm (Evaluation Framework)

To evaluate an algorithm's competitive ratio:

1. Define the input space $\mathcal{I}$ (all possible request sequences).
2. Define $\text{ALG}(I)$ and $\text{OPT}(I)$ for each $I$.
3. Compute the worst-case ratio across all inputs:
   $$
   c = \max_I \frac{\text{ALG}(I)}{\text{OPT}(I)}
   $$
4. Optionally, compute expected competitive ratio for randomized algorithms.

#### Tiny Code (Simulation Example)

Python

```python
def competitive_ratio(alg_func, opt_func, inputs):
    worst_ratio = 0
    for I in inputs:
        alg = alg_func(I)
        opt = opt_func(I)
        ratio = alg / opt if opt > 0 else float('inf')
        worst_ratio = max(worst_ratio, ratio)
    return worst_ratio
```

This framework can simulate multiple inputs to find the empirical ratio.

#### Why It Matters

- The competitive ratio allows theoretical guarantees for *uncertain environments*.
- Helps compare algorithms fairly without assuming probabilistic inputs.
- Bridges theory with systems: LRU, scheduling, caching, and dynamic pricing all rely on it.

#### Try It Yourself

1. Implement online paging, knapsack, or matching.
2. Simulate adversarial inputs to find the worst case.
3. Compute the ratio $\text{ALG}/\text{OPT}$.
4. See how randomization improves the average ratio.

#### Test Cases

| Problem         | Algorithm       | Competitive Ratio |
| --------------- | --------------- | ----------------- |
| Paging          | LRU             | $k$               |
| Online Matching | Ranking         | $1 - 1/e$         |
| Ski Rental      | Rent-then-buy   | $2$               |
| Online Knapsack | Ratio threshold | $1 - 1/e$         |
| Load Balancing  | Greedy          | $2 - 1/m$         |

#### Complexity

Evaluating the ratio often requires:

- Offline optimum via dynamic programming or linear programming
- Online run via simulation
- Worst-case search across generated input sequences

Time complexity depends on the offline algorithm (often $O(n^2)$ or worse).

#### A Gentle Proof (The Big Idea)

For any online algorithm $\text{ALG}$,
the ratio bounds its *relative inefficiency*:

$$
\forall I, \quad \text{ALG}(I) \le c \cdot \text{OPT}(I)
$$

If $c$ is small, $\text{ALG}$ is nearly optimal even against an adversary.
For randomized algorithms, the guarantee is in *expectation*:

$$
E[\text{ALG}(I)] \le c \cdot \text{OPT}(I)
$$

#### Summary Table

| Concept             | Definition                    |
| ------------------- | ----------------------------- |
| Metric              | $\text{ALG}/\text{OPT}$ ratio |
| Purpose             | Quantify online inefficiency  |
| Deterministic Bound | Worst-case ratio              |
| Randomized Bound    | Expected ratio                |
| Ideal Value         | 1 (perfect)                   |
| Common Values       | 2, $1 - 1/e$, $k$             |

The competitive ratio is the philosopher's compass of online algorithms, 
it doesn't promise certainty, but measures *how gracefully* we face the unknown.

### 979. PTAS and FPTAS Schemes (Polynomial-Time Approximation)

Not all optimization problems can be solved efficiently.
Some are NP-hard, meaning no known algorithm can find an exact solution in polynomial time.
But in many real-world scenarios, we don't need the perfect answer, just a *good enough* one.
That's where approximation schemes come in: algorithms that get arbitrarily close to the optimal solution, within controllable precision.

#### What Problem Are We Solving?

Given an optimization problem (usually NP-hard), we want an algorithm that produces a solution whose value is within a small fraction $\varepsilon$ of the optimal one.

For a minimization problem, the goal is:

$$
\frac{\text{ALG}}{\text{OPT}} \le 1 + \varepsilon
$$

For a maximization problem, the goal is:

$$
\frac{\text{ALG}}{\text{OPT}} \ge 1 - \varepsilon
$$

Here, $\varepsilon > 0$ is a user-chosen error tolerance.

#### The Big Picture

There are two main types:

| Type                                                   | Definition                                                                                              | Running Time                       |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| PTAS (Polynomial-Time Approximation Scheme)        | For any $\varepsilon > 0$, runs in time polynomial in $n$, but possibly exponential in $1/\varepsilon$. | $O(n^{f(1/\varepsilon)})$          |
| FPTAS (Fully Polynomial-Time Approximation Scheme) | Runs in time polynomial in both $n$ and $1/\varepsilon$.                                                | $O(\text{poly}(n, 1/\varepsilon))$ |

So FPTAS ⊂ PTAS, it's a stricter and more efficient subclass.

#### Example 1 – Knapsack Problem

For the classic 0/1 Knapsack Problem, we can use an FPTAS by scaling values.

Original problem:

$$
\max \sum_i v_i x_i \quad \text{s.t.} \quad \sum_i w_i x_i \le W, \quad x_i \in {0,1}
$$

We scale down item values by a factor $K = \frac{\varepsilon \cdot V_{\max}}{n}$ and round them:

$$
v_i' = \left\lfloor \frac{v_i}{K} \right\rfloor
$$

Then run the dynamic programming algorithm on these scaled values, which now have a smaller range.
The result guarantees:

$$
(1 - \varepsilon) \cdot \text{OPT} \le \text{ALG} \le \text{OPT}
$$

with runtime:

$$
O\left( \frac{n^3}{\varepsilon} \right)
$$

#### Example 2 – Euclidean Traveling Salesman Problem (TSP)

For points in a plane (Euclidean metric), Arora (1996) developed a PTAS.
By cleverly subdividing the plane into grids and limiting the paths crossing each boundary, it achieves:

$$
\text{ALG} \le (1 + \varepsilon) \cdot \text{OPT}
$$

with runtime:

$$
O(n (\log n)^{O(1/\varepsilon)})
$$

This is a landmark result, TSP is NP-hard, yet admits near-optimal solutions efficiently in geometry.

#### Example 3 – Scheduling Jobs on Machines

Problem: Assign $n$ jobs to $m$ identical machines to minimize the makespan (max load).
A PTAS can be built by:

1. Scheduling the largest $k = O(1/\varepsilon^2)$ jobs optimally by enumeration.
2. Scheduling the rest greedily.

This gives:

$$
\text{ALG} \le (1 + \varepsilon) \cdot \text{OPT}
$$

in polynomial time.

#### Algorithm (Generic Scheme)

```
PTAS(problem, ε):
    for each possible simplified configuration (bounded by f(1/ε)):
        solve reduced subproblem optimally
    return best found solution
```

FPTAS uses a scaling trick to reduce numerical precision so dynamic programming remains polynomial in $1/ε$.

#### Tiny Code (FPTAS for Knapsack)

Python

```python
def knapsack_fptas(values, weights, W, ε):
    n = len(values)
    vmax = max(values)
    K = ε * vmax / n
    scaled = [int(v / K) for v in values]

    Vsum = sum(scaled)
    dp = [float('inf')] * (Vsum + 1)
    dp[0] = 0

    for i in range(n):
        for v in range(Vsum, scaled[i] - 1, -1):
            dp[v] = min(dp[v], dp[v - scaled[i]] + weights[i])

    for v in range(Vsum, -1, -1):
        if dp[v] <= W:
            return v * K
```

#### Why It Matters

- PTAS and FPTAS let us solve *impossible* problems *practically*.
- They offer a tradeoff between speed and accuracy.
- Widely used in optimization, operations research, AI planning, and data science.
- They provide the bridge between theory (NP-hardness) and practice (good solutions fast).

#### Try It Yourself

1. Run the FPTAS on small knapsack instances.
2. Change $\varepsilon$ and observe how runtime and quality change.
3. Compare with the exact DP solution.
4. See how small $\varepsilon$ quickly increases computation.

#### Test Cases

| Problem        | Scheme               | Guarantee           | Type  |
| -------------- | -------------------- | ------------------- | ----- |
| Knapsack       | Scaling DP           | $(1 - \varepsilon)$ | FPTAS |
| Euclidean TSP  | Arora (1996)         | $(1 + \varepsilon)$ | PTAS  |
| Job Scheduling | Enumeration + Greedy | $(1 + \varepsilon)$ | PTAS  |
| Bin Packing    | Rounding + First-Fit | $(1 + \varepsilon)$ | PTAS  |

#### Complexity

| Scheme                 | Time                               | Example Problem  |
| ---------------------- | ---------------------------------- | ---------------- |
| PTAS                   | $O(n^{f(1/\varepsilon)})$          | TSP              |
| FPTAS                  | $O(\text{poly}(n, 1/\varepsilon))$ | Knapsack         |
| Constant Approximation | $O(n)$                             | Greedy Set Cover |

#### A Gentle Proof (Why It Works)

Approximation schemes rely on rounding and bounding.
Let $\text{OPT}$ be the optimal value, and $\text{ALG}$ the approximate one.

If each item's value is rounded to within $(1 - \varepsilon)$ of the true value:

$$
\text{ALG} \ge (1 - \varepsilon) \cdot \text{OPT}
$$

Since rounding errors compound linearly, not exponentially,
precision scales gracefully with $\varepsilon$, maintaining polynomial runtime.

#### Summary Table

| Property  | PTAS                            | FPTAS                    |
| --------- | ------------------------------- | ------------------------ |
| Accuracy  | Arbitrary $\varepsilon$         | Arbitrary $\varepsilon$  |
| Time      | Poly($n$), exp($1/\varepsilon$) | Poly($n, 1/\varepsilon$) |
| Guarantee | $(1 \pm \varepsilon)$           | $(1 \pm \varepsilon)$    |
| Example   | Euclidean TSP                   | Knapsack                 |
| Use Case  | Theoretical & geometric         | Practical numeric        |

Approximation schemes remind us that perfect is often impractical, 
but almost perfect can be beautifully efficient.

### 980. Primal–Dual Method (Approximate Combinatorial Optimization)

The Primal–Dual Method is a powerful framework for designing approximation algorithms for NP-hard problems.
Instead of solving optimization problems exactly, it builds both the primal and dual linear programs together, maintaining feasible or nearly feasible solutions, and stopping when a balance is reached between cost and coverage.

#### What Problem Are We Solving?

Many combinatorial optimization problems can be expressed as linear programs (LPs).

A primal LP might look like:

$$
\min c^T x \quad \text{s.t.} \quad A x \ge b, ; x \ge 0
$$

Its dual LP is:

$$
\max b^T y \quad \text{s.t.} \quad A^T y \le c, ; y \ge 0
$$

The Primal–Dual Method constructs solutions for both simultaneously, maintaining relationships between $x$ and $y$ so that their costs remain close.

This gives approximation guarantees even when we can't solve the LP exactly.

#### Intuitive Idea

Think of the primal and dual as two players:

- The primal player selects elements (edges, sets, facilities) to satisfy constraints.
- The dual player raises prices or penalties for unsatisfied constraints.

The algorithm iteratively raises dual variables until some constraint becomes "tight,"
then adds the corresponding primal variable (e.g., selects an edge, opens a facility).
This process repeats until all constraints are satisfied.

The ratio between total primal cost and total dual value gives the approximation factor.

#### Example – Vertex Cover Problem

Goal: Choose the smallest set of vertices covering all edges in a graph $G = (V, E)$.

Primal (minimization):

$$
\begin{aligned}
\text{minimize} \quad & \sum_{v \in V} x_v \
\text{subject to} \quad & x_u + x_v \ge 1 \quad \forall (u,v) \in E \
& x_v \ge 0
\end{aligned}
$$

Dual (maximization):

$$
\begin{aligned}
\text{maximize} \quad & \sum_{(u,v) \in E} y_{uv} \
\text{subject to} \quad & \sum_{(u,v): v \in (u,v)} y_{uv} \le 1 \quad \forall v \in V \
& y_{uv} \ge 0
\end{aligned}
$$

Algorithm (Primal–Dual):

1. Initialize all $y_{uv} = 0$ and all edges uncovered.
2. Increase $y_{uv}$ for uncovered edges until some vertex constraint becomes tight (sum of incident $y_{uv} = 1$).
3. Add that vertex to the cover (set $x_v = 1$).
4. Repeat until all edges are covered.

At the end:

- Dual cost = total raised $y_{uv}$
- Primal cost = number of selected vertices

This algorithm yields a 2-approximation, since each edge can "charge" at most twice.

#### Example – Set Cover Problem

Sets $S_1, S_2, \dots, S_m$ with costs $c_i$, need to cover all elements $U$.

Primal:

$$
\min \sum_i c_i x_i \quad \text{s.t.} \quad \sum_{i: e \in S_i} x_i \ge 1, ; x_i \ge 0
$$

Dual:

$$
\max \sum_{e \in U} y_e \quad \text{s.t.} \quad \sum_{e \in S_i} y_e \le c_i, ; y_e \ge 0
$$

Algorithm Sketch:

1. Start with $y_e = 0$.
2. Uniformly raise $y_e$ for uncovered elements until some set constraint becomes tight ($\sum_{e \in S_i} y_e = c_i$).
3. Pick that set $S_i$ into the cover.
4. Repeat until all elements are covered.

Approximation ratio:
$$
H_n = 1 + \frac{1}{2} + \frac{1}{3} + \dots + \frac{1}{n} = O(\log n)
$$
Same as the greedy algorithm.

#### Tiny Code (Simplified Vertex Cover Example)

Python

```python
def primal_dual_vertex_cover(edges, n):
    y = {e: 0 for e in edges}
    cover = set()
    while edges:
        (u, v) = edges.pop()
        if u not in cover and v not in cover:
            cover.add(u)
            cover.add(v)
    return cover
```

This simple primal-dual interpretation of "cover both endpoints" yields a 2-approximation.

#### Why It Matters

- Offers a constructive, combinatorial way to derive approximation bounds.
- Avoids solving LPs directly while still exploiting LP structure.
- Yields many of the best-known results for NP-hard problems:

  * 2-approximation for Vertex Cover
  * $O(\log n)$ for Set Cover
  * Constant-factor approximations for Facility Location and Steiner Tree

#### Try It Yourself

1. Derive primal and dual forms of the Facility Location problem.
2. Implement a primal-dual algorithm that raises client "prices."
3. Track when facility opening constraints become tight.
4. Compare the solution cost to the LP relaxation.

#### Test Cases

| Problem           | Approximation | Method                      |
| ----------------- | ------------- | --------------------------- |
| Vertex Cover      | 2             | Tightness-based primal-dual |
| Set Cover         | $O(\log n)$   | Dual raising                |
| Facility Location | 1.61          | Jain–Vazirani primal-dual   |
| Steiner Tree      | 2             | Edge-growing                |

#### Complexity

- Time: $O(|E| + |V|)$ for graphs, or polynomial for LP-structured problems
- Space: $O(|V|)$
- Approximation Ratio: Problem-dependent, usually constant or logarithmic

#### A Gentle Proof (Why It Works)

From weak duality:

$$
b^T y \le c^T x
$$

At each step, the dual value grows until a constraint becomes tight, ensuring:

$$
c^T x \le \alpha \cdot b^T y
$$

for some $\alpha$, the approximation factor.
This holds because each primal constraint is satisfied once its corresponding dual variable stops increasing.

For example, in Vertex Cover, $\alpha = 2$ since each edge can contribute to two vertices.

#### Summary Table

| Concept         | Description                                    |
| --------------- | ---------------------------------------------- |
| Framework       | Builds primal and dual LPs together            |
| Strategy        | Raise duals until primal constraint tight      |
| Guarantee       | $\text{cost(ALG)} \le \alpha \cdot \text{OPT}$ |
| Common $\alpha$ | 2 (VC), $O(\log n)$ (SC)                       |
| Advantage       | No need to solve LP directly                   |

The Primal–Dual Method is the quiet backbone of approximation theory, 
balancing two mirrors of the same problem until their reflection yields near-optimal beauty.

# Section 99. Fairness, Causal Inference, and Robust Optimization 

### 981. Reweighting for Fairness

Reweighting is a simple but powerful algorithmic strategy for reducing bias in machine learning models.
It adjusts the *importance weights* of training samples so that different demographic or sensitive groups contribute *equally* to the learning process.

This forms the foundation of *pre-processing fairness methods*, adjusting data before training to correct imbalances in representation or outcomes.

#### What Problem Are We Solving?

Real-world datasets often reflect *historical bias*.
For example, a dataset for loan approval might contain fewer positive examples for a particular group due to systemic discrimination.
Training directly on such data leads to unfair predictions.

We want a method that preserves accuracy while making model outcomes more equitable across groups (e.g., gender, race, or age).

#### Basic Idea

If data is biased, certain combinations of group membership and label are underrepresented.
Reweighting aims to correct that imbalance.

Each instance gets a weight:

$$
w(x, a, y) = \frac{P(A=a) , P(Y=y)}{P(A=a, Y=y)}
$$

Where:

- $A$ = protected attribute (e.g., gender)
- $Y$ = true label (e.g., loan approved)
- $P(A=a, Y=y)$ = joint distribution in the observed data
- $P(A=a) P(Y=y)$ = product of marginals, what we'd expect if $A$ and $Y$ were independent

This adjustment *breaks the dependency* between sensitive attributes and labels in the training data.

#### How It Works (Plain Language)

1. Count how often each pair $(A=a, Y=y)$ appears in the data.
2. Compute expected frequencies under independence: $P(A=a) P(Y=y)$.
3. Compute reweighting factor $w(a, y)$ = ratio of expected to actual frequency.
4. Apply these weights when training the model, each example's loss contribution is scaled by its weight.

This ensures that:
$$
A \perp Y \quad \text{(statistical independence enforced in expectation)}
$$

#### Example

Suppose:

| Group    | Label = 1 | Label = 0 | Total |
| -------- | --------- | --------- | ----- |
| A=Male   | 700       | 300       | 1000  |
| A=Female | 300       | 700       | 1000  |

Marginals:

- $P(A=\text{Male}) = P(A=\text{Female}) = 0.5$
- $P(Y=1) = 0.5$, $P(Y=0) = 0.5$

Then the ideal joint distribution under independence is $0.25$ for each $(A,Y)$ combination.
Observed probabilities:

| (A,Y)      | Observed | Expected | Weight |
| ---------- | -------- | -------- | ------ |
| (Male,1)   | 0.35     | 0.25     | 0.714  |
| (Male,0)   | 0.15     | 0.25     | 1.667  |
| (Female,1) | 0.15     | 0.25     | 1.667  |
| (Female,0) | 0.35     | 0.25     | 0.714  |

During training, each instance is weighted accordingly.
This makes male and female contributions balanced with respect to positive and negative outcomes.

#### Tiny Code

Python (using sklearn)

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight

# Example data
A = np.array(['M', 'M', 'F', 'F', 'M', 'F'])
Y = np.array([1, 0, 1, 0, 1, 0])

# Compute reweighting factors
unique_A, unique_Y = np.unique(A), np.unique(Y)
weights = np.zeros(len(A))

for a in unique_A:
    for y in unique_Y:
        idx = np.where((A == a) & (Y == y))[0]
        pa = np.mean(A == a)
        py = np.mean(Y == y)
        pay = len(idx) / len(A)
        w = (pa * py) / pay
        weights[idx] = w

# Train weighted model
clf = LogisticRegression()
clf.fit(np.ones((len(A), 1)), Y, sample_weight=weights)
```

#### Why It Matters

- It reduces bias before model training (pre-processing).
- Works with any classifier that supports weighted samples.
- Maintains interpretability and control, easy to explain and audit.
- Forms the foundation for more advanced methods like Adversarial Debiasing or Fair Reweighting (Zafar et al.).

#### Try It Yourself

1. Compute reweighting factors for your dataset.
2. Compare model accuracy before and after reweighting.
3. Evaluate fairness metrics (e.g., demographic parity difference, equal opportunity).
4. Tune $\varepsilon$ thresholds for acceptable fairness–accuracy tradeoff.

#### Test Cases

| Dataset       | Fairness Metric    | Before | After |
| ------------- | ------------------ | ------ | ----- |
| Adult Income  | Demographic Parity | 0.23   | 0.05  |
| COMPAS        | Equal Opportunity  | 0.18   | 0.07  |
| Loan Approval | Statistical Parity | 0.20   | 0.04  |

#### Complexity

| Step                          | Time             | Space  |
| ----------------------------- | ---------------- | ------ |
| Count group-label frequencies | $O(n)$           | $O(k)$ |
| Compute weights               | $O(k)$           | $O(k)$ |
| Weighted training             | Depends on model |,      |

Usually negligible overhead compared to training time.

#### A Gentle Proof (Why It Works)

The fairness correction relies on making the joint $(A, Y)$ distribution factorize into marginals:

$$
P(A, Y) = P(A)P(Y)
$$

By assigning weights:

$$
w(a, y) = \frac{P(A=a)P(Y=y)}{P(A=a, Y=y)}
$$

the weighted empirical distribution $\hat{P}_w$ satisfies:

$$
\hat{P}_w(A=a, Y=y) = P(A=a) P(Y=y)
$$

Hence, any model minimizing weighted loss learns under a fairer, balanced distribution.

#### Summary Table

| Concept   | Description                                           |
| --------- | ----------------------------------------------------- |
| Goal      | Remove dependency between $A$ (group) and $Y$ (label) |
| Formula   | $w(a,y) = \frac{P(A=a) P(Y=y)}{P(A=a, Y=y)}$          |
| Type      | Pre-processing fairness                               |
| Advantage | Simple, model-agnostic                                |
| Guarantee | Demographic parity (approximate)                      |

Reweighting is fairness through *rebalancing*:
not changing the world, just changing the lens through which the model sees it.

### 982. Demographic Parity Constraint

Demographic Parity (DP), also called Statistical Parity, is one of the most fundamental fairness criteria in machine learning.
It ensures that *predicted outcomes* are independent of *sensitive attributes* such as gender, race, or age.

In simple terms: the model should give positive outcomes at the same rate for all groups.

#### What Problem Are We Solving?

Even when models are accurate, they can produce biased predictions.
For example:

- A loan approval model may approve 80% of male applicants but only 40% of female applicants.
- A hiring algorithm may favor younger candidates, even with equal qualifications.

Demographic Parity seeks to eliminate these differences by constraining the model to produce equal acceptance rates across groups.

#### The Fairness Condition

Let:

- $A$ = sensitive attribute (e.g., gender)
- $\hat{Y}$ = model's predicted label (e.g., approved = 1, denied = 0)

Then Demographic Parity requires:

$$
P(\hat{Y} = 1 \mid A = 0) = P(\hat{Y} = 1 \mid A = 1)
$$

That is, the probability of a positive outcome should be the same, regardless of $A$.
In practice, we relax it slightly to allow a tolerance $\varepsilon$:

$$
\left| P(\hat{Y} = 1 \mid A = 0) - P(\hat{Y} = 1 \mid A = 1) \right| \le \varepsilon
$$

#### Intuitive Example

Suppose we train a classifier on job applications.
Outcomes before applying DP constraint:

| Group  | Positive Rate |
| ------ | ------------- |
| Male   | 0.70          |
| Female | 0.45          |

DP requires adjusting the model or thresholds so both groups achieve roughly equal positive rates (say 0.57 ± 0.02).

We can do this by:

- Changing decision thresholds per group, or
- Adding a constraint in the loss function during training.

#### Loss with Demographic Parity Penalty

The standard empirical loss is:

$$
L = \frac{1}{n} \sum_i \ell(f(x_i), y_i)
$$

We add a fairness penalty:

$$
L_{\text{fair}} = L + \lambda , \left| , E[\hat{Y} | A = 0] - E[\hat{Y} | A = 1] , \right|
$$

where $\lambda$ controls the tradeoff between accuracy and fairness.
A higher $\lambda$ enforces stronger fairness at the cost of model performance.

#### Tiny Code (Fair Logistic Regression with DP Penalty)

Python (simplified)

```python
import torch
import torch.nn as nn
import torch.optim as optim

class FairLogReg(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Linear(d, 1)

    def forward(self, x):
        return torch.sigmoid(self.w(x))

def demographic_parity_penalty(pred, A):
    # Mean positive rate per group
    p0 = pred[A == 0].mean()
    p1 = pred[A == 1].mean()
    return torch.abs(p0 - p1)

# Training loop
model = FairLogReg(d=5)
opt = optim.Adam(model.parameters(), lr=0.01)
λ = 1.0

for x, y, A in data_loader:
    pred = model(x).squeeze()
    loss = nn.BCELoss()(pred, y)
    dp_penalty = demographic_parity_penalty(pred, A)
    total_loss = loss + λ * dp_penalty
    opt.zero_grad()
    total_loss.backward()
    opt.step()
```

This model learns while actively penalizing group-level imbalance in prediction rates.

#### Why It Matters

- Prevents discrimination *in outcomes*, not just in labels.
- Serves as a first-line fairness constraint in many systems (e.g., credit scoring, advertising).
- Forms the foundation of other criteria like Equalized Odds and Predictive Parity.
- Works well when group membership is known and relevant for fairness analysis.

#### Try It Yourself

1. Train a standard logistic regression model.
2. Measure $P(\hat{Y}=1 \mid A=0)$ and $P(\hat{Y}=1 \mid A=1)$.
3. Add a fairness penalty (or threshold adjustment).
4. Observe how accuracy and fairness trade off.
5. Tune $\lambda$ to balance fairness and utility.

#### Test Cases

| Dataset        | Metric                | Before | After DP Constraint |
| -------------- | --------------------- | ------ | ------------------- |
| Adult Income   | $\Delta P(\hat{Y}=1)$ | 0.24   | 0.05                |
| COMPAS         | $\Delta P(\hat{Y}=1)$ | 0.18   | 0.07                |
| Bank Marketing | $\Delta P(\hat{Y}=1)$ | 0.21   | 0.03                |

#### Complexity

| Step                | Time                    | Space  |
| ------------------- | ----------------------- | ------ |
| Compute group means | $O(n)$                  | $O(k)$ |
| Add penalty to loss | $O(1)$                  |,      |
| Training cost       | Slightly above baseline |,      |

Fairness enforcement adds negligible computational overhead.

#### A Gentle Proof (Why It Works)

If $\hat{Y}$ is trained to minimize

$$
L_{\text{fair}} = L + \lambda , | E[\hat{Y}|A=0] - E[\hat{Y}|A=1] |
$$

then at equilibrium:

$$
E[\hat{Y}|A=0] \approx E[\hat{Y}|A=1]
$$

which implies approximate independence:

$$
\hat{Y} \perp A
$$

Thus, group membership no longer influences outcomes, the core fairness condition for Demographic Parity.

#### Summary Table

| Concept           | Description                                      |                    |       |
| ----------------- | ------------------------------------------------ | ------------------ | ----- |
| Fairness Type     | Demographic Parity (Statistical Parity)          |                    |       |
| Mathematical Form | $P(\hat{Y}=1                                     | A=0) = P(\hat{Y}=1 | A=1)$ |
| Implementation    | Penalty or post-processing                       |                    |       |
| Strength          | Group-level equality                             |                    |       |
| Limitation        | May reduce accuracy or ignore label correlations |                    |       |

Demographic Parity is fairness in its purest form, 
it doesn't ask *why* outcomes differ, only that they *shouldn't* differ at all.

### 983. Equalized Odds

Equalized Odds (EO) is a fairness criterion that goes deeper than *Demographic Parity*.
Instead of just equalizing overall prediction rates, it demands equality conditioned on the true outcome, ensuring that a model is *equally accurate* (and equally mistaken) across demographic groups.

It's a fairness definition that focuses on error balance, not just outcome rates.

#### What Problem Are We Solving?

Demographic Parity ensures equal positive rates but ignores whether those predictions are correct.
A model could trivially satisfy DP by flipping coins, giving random approvals equally across groups.
That's fair in form, but not in truth.

Equalized Odds fixes this by enforcing that both *true positive rates* and *false positive rates* are equal across groups.

#### The Fairness Condition

Let:

- $A$ = sensitive attribute (e.g., gender)
- $Y$ = true label
- $\hat{Y}$ = predicted label

Then Equalized Odds requires:

$$
P(\hat{Y} = 1 \mid Y = y, A = 0) = P(\hat{Y} = 1 \mid Y = y, A = 1)
\quad \text{for } y \in {0, 1}
$$

That is:

- Equal true positive rates (TPR) across groups
- Equal false positive rates (FPR) across groups

In practice, we measure:
$$
\text{TPR gap} = | P(\hat{Y}=1|Y=1,A=0) - P(\hat{Y}=1|Y=1,A=1) |
$$
$$
\text{FPR gap} = | P(\hat{Y}=1|Y=0,A=0) - P(\hat{Y}=1|Y=0,A=1) |
$$

Both should be small, ideally below a threshold $\varepsilon$.

#### Intuitive Example

Suppose we build a medical diagnosis model.

| Group  | True Positive Rate | False Positive Rate |
| ------ | ------------------ | ------------------- |
| Male   | 0.85               | 0.10                |
| Female | 0.70               | 0.05                |

The model is better at catching positives for males.
Equalized Odds would require adjustments (e.g., shifting decision thresholds) so both groups have approximately equal TPR and FPR, say 0.78 and 0.07 respectively.

#### Implementation Strategies

There are three main ways to enforce Equalized Odds:

1. Pre-processing
   Modify data or sample weights so that errors are balanced across groups (e.g., reweighting, resampling).

2. In-processing
   Add fairness regularizers to the loss:
   $$
   L_{\text{fair}} = L + \lambda_1 | \text{TPR}_0 - \text{TPR}_1 | + \lambda_2 | \text{FPR}_0 - \text{FPR}_1 |
   $$

3. Post-processing
   Adjust decision thresholds per group after training to equalize error rates (Hardt et al., 2016).

#### Tiny Code (Threshold Adjustment for EO)

Python (post-processing)

```python
import numpy as np

def equalized_odds_thresholds(y_true, y_pred, A):
    thresholds = {}
    for a in np.unique(A):
        yg = y_pred[A == a]
        tg = y_true[A == a]
        # Compute best threshold to equalize TPR/FPR
        best_t, best_gap = 0.5, 1.0
        for t in np.linspace(0, 1, 101):
            pred_bin = (yg >= t).astype(int)
            tpr = np.mean(pred_bin[tg == 1])
            fpr = np.mean(pred_bin[tg == 0])
            gap = abs(tpr - fpr)
            if gap < best_gap:
                best_t, best_gap = t, gap
        thresholds[a] = best_t
    return thresholds
```

This computes per-group thresholds that minimize error imbalance.

#### Why It Matters

- Ensures equal treatment in correctness, not just outcome rates.
- Ideal for high-stakes domains like healthcare, justice, or hiring.
- Avoids "fair but useless" models (unlike Demographic Parity).
- Provides a tradeoff between fairness and accuracy that's more ethically meaningful.

#### Try It Yourself

1. Train any binary classifier and record predictions per group.
2. Compute TPR and FPR for each group.
3. Adjust thresholds or weights to minimize both gaps.
4. Compare model fairness (EO gaps) and accuracy before and after.

#### Test Cases

| Dataset      | Metric  | Before | After Equalized Odds |
| ------------ | ------- | ------ | -------------------- |
| COMPAS       | TPR gap | 0.22   | 0.05                 |
| COMPAS       | FPR gap | 0.18   | 0.04                 |
| Adult Income | TPR gap | 0.15   | 0.03                 |
| Adult Income | FPR gap | 0.12   | 0.02                 |

#### Complexity

| Step                             | Time           | Space  |
| -------------------------------- | -------------- | ------ |
| Threshold search                 | $O(k \cdot n)$ | $O(k)$ |
| Compute rates                    | $O(n)$         | $O(1)$ |
| Training penalty (in-processing) | Minor          |,      |

Here $k$ is the number of candidate thresholds.

#### A Gentle Proof (Why It Works)

Let $\hat{Y}_A$ denote predictions per group.
By adjusting thresholds to make $P(\hat{Y}=1|Y=y,A=a)$ equal across $A$, we enforce:

$$
\hat{Y} \perp A \mid Y
$$

This conditional independence expresses fairness:
the model's errors no longer depend on group membership once true outcomes are known.

#### Summary Table

| Concept        | Description                           |
| -------------- | ------------------------------------- |
| Fairness Type  | Equalized Odds                        |
| Condition      | $\hat{Y} \perp A \mid Y$              |
| Constraints    | Equal TPR and FPR across groups       |
| Implementation | Pre-, In-, or Post-processing         |
| Benefit        | Ensures equal *error behavior*        |
| Limitation     | May require group-specific thresholds |

Equalized Odds is fairness with awareness, 
not pretending everyone's the same, but ensuring everyone is *treated equally in being right or wrong.*

### 984. Adversarial Debiasing

Adversarial Debiasing is one of the most elegant and powerful methods to achieve fairness in machine learning.
It uses the same philosophy that drives Generative Adversarial Networks (GANs): two models compete, one tries to make accurate predictions, while the other tries to detect unfair bias.
Through this tension, the predictor learns to *hide* information about sensitive attributes, resulting in fairer outcomes.

#### What Problem Are We Solving?

Many learning algorithms, even when trained on reweighted or balanced data, still leak information about sensitive attributes ($A$), such as gender or race.
The model might not use $A$ explicitly, but it can *infer* it from correlated features (like zip code or occupation).

We want to train a predictor $f_\theta(x)$ that:

1. Predicts the target $Y$ accurately, and
2. Produces predictions $\hat{Y}$ that contain as little information about $A$ as possible.

#### The Core Idea

We build two networks (or modules):

1. Predictor (Main model)
   Learns $f_\theta(x)$ to predict the true label $Y$.

2. Adversary (Fairness discriminator)
   Learns $g_\phi(\hat{Y})$ (or sometimes $g_\phi(h)$ using internal representations)
   to predict the sensitive attribute $A$ from the predictor's output.

The predictor is trained to *minimize prediction loss* while simultaneously *maximizing the adversary's loss*, making it hard for the adversary to recover $A$.

#### The Objective

Let $L_y$ be the prediction loss (e.g., cross-entropy for $Y$),
and $L_a$ the adversary loss (for predicting $A$).

We optimize a min–max objective:

$$
\min_{\theta} \max_{\phi} , \big( L_y(f_\theta(x), y) - \lambda L_a(g_\phi(f_\theta(x)), a) \big)
$$

where $\lambda$ controls the fairness–accuracy tradeoff.

- The predictor wants to minimize $L_y$ and maximize $L_a$ (fool the adversary).
- The adversary wants to minimize $L_a$ (predict $A$ as well as possible).

When the system reaches equilibrium:
$$
\hat{Y} \perp A
$$
i.e., predictions no longer carry information about group membership.

#### Tiny Code (Adversarial Debiasing Framework)

PyTorch-style pseudocode

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Predictor model (main task)
class Predictor(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())

    def forward(self, x):
        return self.net(x)

# Adversary model (fairness discriminator)
class Adversary(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 8), nn.ReLU(), nn.Linear(8, 1), nn.Sigmoid())

    def forward(self, y_pred):
        return self.net(y_pred)

predictor = Predictor(d=10)
adversary = Adversary()

opt_pred = optim.Adam(predictor.parameters(), lr=0.001)
opt_adv = optim.Adam(adversary.parameters(), lr=0.001)
λ = 1.0

for x, y, a in data_loader:
    y_pred = predictor(x)

    # Train adversary
    a_pred = adversary(y_pred.detach())
    loss_a = nn.BCELoss()(a_pred, a.float())
    opt_adv.zero_grad()
    loss_a.backward()
    opt_adv.step()

    # Train predictor
    y_pred = predictor(x)
    a_pred = adversary(y_pred)
    loss_y = nn.BCELoss()(y_pred, y.float())
    loss_total = loss_y - λ * nn.BCELoss()(a_pred, a.float())
    opt_pred.zero_grad()
    loss_total.backward()
    opt_pred.step()
```

This loop alternates between strengthening the adversary and weakening its influence on the predictor, balancing fairness and accuracy dynamically.

#### Why It Matters

- Learns representation-level fairness, deeper than reweighting or thresholding.
- Works even when sensitive information is implicit or correlated.
- Applicable to text, vision, tabular, or graph data.
- Enables flexible tradeoff via $\lambda$.
- Inspired by information theory: reduce mutual information between $\hat{Y}$ and $A$.

#### Try It Yourself

1. Start with a dataset like Adult Income (with `sex` as $A$).
2. Train a baseline classifier, measure demographic disparity.
3. Add an adversarial head to predict $A$.
4. Tune $\lambda$:

   * $\lambda = 0$ → maximum accuracy, no fairness.
   * $\lambda > 1$ → strong fairness, possible accuracy drop.
5. Observe tradeoffs in fairness metrics.

#### Test Cases

| Dataset       | Metric             | Before | After Adversarial Debiasing |
| ------------- | ------------------ | ------ | --------------------------- |
| Adult Income  | Demographic Parity | 0.23   | 0.05                        |
| COMPAS        | Equalized Odds Gap | 0.21   | 0.07                        |
| German Credit | TPR Gap            | 0.19   | 0.06                        |

#### Complexity

| Step             | Time                                   | Space  |
| ---------------- | -------------------------------------- | ------ |
| Predictor update | $O(n d)$                               | $O(d)$ |
| Adversary update | $O(n)$                                 | $O(1)$ |
| Total            | Slightly slower than standard training |,      |

Training alternates between two optimizers, typically doubles runtime, but remains polynomial.

#### A Gentle Proof (Why It Works)

At equilibrium of the min–max game:
$$
\nabla_\phi L_a = 0, \quad \nabla_\theta (L_y - \lambda L_a) = 0
$$

This implies the adversary cannot extract information about $A$ from $\hat{Y}$.
Hence the mutual information $I(\hat{Y}; A)$ is minimized:
$$
I(\hat{Y}; A) \approx 0
$$

so predictions become *statistically independent* of the sensitive attribute, the formal definition of fairness under demographic parity.

#### Summary Table

| Concept       | Description                                        |
| ------------- | -------------------------------------------------- |
| Fairness Type | In-processing (Adversarial)                        |
| Objective     | $\min_\theta \max_\phi (L_y - \lambda L_a)$        |
| Mechanism     | Adversary tries to detect bias; predictor hides it |
| Key Idea      | Fairness through confusion                         |
| Strength      | Learns fair latent representations                 |
| Limitation    | Requires adversary tuning and joint stability      |

Adversarial Debiasing is fairness by competition, 
two networks locked in a game where justice emerges from balance.

### 985. Causal DAG Discovery

Causal Directed Acyclic Graph (DAG) Discovery is the process of uncovering cause-and-effect relationships from data.
Unlike correlation-based learning, causal discovery tries to answer *what happens if we intervene*, going beyond prediction into the structure of reality itself.

#### What Problem Are We Solving?

Machine learning models often find associations:

- "Smoking and yellow teeth are correlated."
  But only causal analysis can tell us:
- "Smoking causes yellow teeth."

Causal discovery formalizes this by identifying a graph structure $G = (V, E)$ over variables $V$, where edges $X_i \to X_j$ represent direct causal influence.

#### Causal Graph Basics

A causal DAG is a directed acyclic graph where:

- Each node represents a variable.
- Each edge represents a causal relation ($X_i \to X_j$ means $X_i$ directly causes $X_j$).
- No directed cycles exist (no feedback loops).

We assume data follows the Causal Markov Condition:
$$
P(X_1, \dots, X_n) = \prod_i P(X_i \mid \text{Pa}(X_i))
$$
where $\text{Pa}(X_i)$ are the parents (direct causes) of $X_i$ in the graph.

#### Two Major Approaches

1. Constraint-Based (Conditional Independence)

   * Tests statistical independence among variables.
   * Builds edges consistent with those tests.
   * Example: PC Algorithm (Peter–Clark).

2. Score-Based (Optimization)

   * Assigns a score to each DAG based on how well it fits the data (e.g., BIC score).
   * Searches over DAG space to maximize score.
   * Example: GES (Greedy Equivalence Search).

#### The PC Algorithm (Plain Overview)

Input: dataset with $n$ variables.
Goal: build DAG capturing causal dependencies.

| Step | Description                                                                                                      |
| ---- | ---------------------------------------------------------------------------------------------------------------- |
| 1    | Start with a fully connected undirected graph.                                                                   |
| 2    | For each pair $(X_i, X_j)$, test if they are conditionally independent given some subset $S$ of other variables. |
| 3    | If independent, remove the edge between them.                                                                    |
| 4    | Orient remaining edges using logical rules (e.g., collider rules).                                               |
| 5    | Output resulting partially directed DAG (CPDAG).                                                                 |

This process combines statistics and logic to infer causal directions.

#### Tiny Code (PC Algorithm Simplified)

Python pseudocode (using partial correlations)

```python
import itertools
import numpy as np
from scipy.stats import pearsonr

def cond_independent(x, y, cond, data, alpha=0.05):
    if not cond:
        r, _ = pearsonr(data[:, x], data[:, y])
    else:
        # Simple linear regression residual method
        Xc = data[:, cond]
        beta_x = np.linalg.lstsq(Xc, data[:, x], rcond=None)[0]
        beta_y = np.linalg.lstsq(Xc, data[:, y], rcond=None)[0]
        rx = data[:, x] - Xc @ beta_x
        ry = data[:, y] - Xc @ beta_y
        r, _ = pearsonr(rx, ry)
    return abs(r) < alpha

def pc_algorithm(data):
    n_vars = data.shape[1]
    adj = np.ones((n_vars, n_vars)) - np.eye(n_vars)
    for l in range(n_vars - 2):
        for (i, j) in itertools.combinations(range(n_vars), 2):
            if adj[i, j]:
                for cond in itertools.combinations([k for k in range(n_vars) if k not in [i, j]], l):
                    if cond_independent(i, j, cond, data):
                        adj[i, j] = adj[j, i] = 0
                        break
    return adj
```

This simplified version removes edges that show conditional independence.

#### Why It Matters

- Reveals *mechanistic* understanding, not just correlation.
- Essential for reasoning under intervention: "What if we change $X$?"
- Foundation for fairness-aware models and causal inference in AI.
- Helps prevent spurious associations that mislead decisions.

#### Try It Yourself

1. Generate synthetic data:
   $$
   X \to Y, \quad Z = X + \epsilon, \quad Y = X + Z + \text{noise}
   $$
2. Run a causal discovery algorithm (e.g., PC or GES).
3. Check if it recovers the original causal directions.
4. Compare with correlations, note where they differ.

#### Test Cases

| Dataset            | True Causal Structure    | Discovered Edges (PC)             |
| ------------------ | ------------------------ | --------------------------------- |
| Synthetic (3 vars) | $X \to Y \to Z$          | $X \to Y$, $Y \to Z$              |
| Linear Gaussian    | $A \to B$, $A \to C$     | $A \to B$, $A \to C$              |
| Nonlinear          | $X \to Y$, $Y \not\to X$ | Partial DAG (direction ambiguous) |

#### Complexity

| Step               | Time     | Space    |
| ------------------ | -------- | -------- |
| Independence tests | $O(n^k)$ | $O(n^2)$ |
| Edge orientation   | $O(n^2)$ | $O(n^2)$ |

where $k$ is the maximum conditioning set size (typically small).
Causal discovery scales poorly with dimensionality but can be optimized using sparsity assumptions.

#### A Gentle Proof (Why It Works)

Under the Causal Markov Condition and Faithfulness Assumption:

- If $X_i$ and $X_j$ are conditionally independent given $S$,
  then there is no direct edge between $X_i$ and $X_j$.

Thus, constraint-based methods can recover the Markov equivalence class of the true DAG.
That is, all DAGs that encode the same conditional independencies.

#### Summary Table

| Concept         | Description                                        |
| --------------- | -------------------------------------------------- |
| Framework       | Causal Inference                                   |
| Goal            | Discover causal structure (DAG) from data          |
| Key Assumptions | Markov + Faithfulness                              |
| Main Methods    | PC (constraint), GES (score), NOTEARS (continuous) |
| Output          | Causal DAG or CPDAG                                |
| Limitation      | Cannot always orient all edges uniquely            |

Causal DAG Discovery teaches algorithms *why* things happen,
not just *when* they co-occur, turning data into cause, not coincidence.

### 986. Propensity Score Matching

Propensity Score Matching (PSM) is a cornerstone technique in causal inference, a way to simulate randomized experiments from observational data.
It balances treatment and control groups by matching samples that have similar probabilities of receiving treatment, given observed covariates.

The key idea: if two individuals have the same *propensity* to receive a treatment, any difference in outcomes between them can be attributed to the treatment itself.

#### What Problem Are We Solving?

When treatment assignment isn't random, comparing treated vs. untreated groups directly gives biased results.
For example, in a medical study:

- Healthier patients may be more likely to receive a treatment.
- So the observed outcome difference reflects both the treatment *and* their health status.

We need a way to control for confounding variables $X$ to estimate the true *causal effect* of treatment $T$ on outcome $Y$.

#### The Core Idea

Define the propensity score as the probability of receiving treatment given the covariates:
$$
e(x) = P(T = 1 \mid X = x)
$$

The method proceeds in three steps:

1. Estimate $e(x)$, usually with logistic regression or an ML classifier.
2. Match treated and untreated samples with similar $e(x)$.
3. Compare outcomes $Y$ between matched pairs to estimate treatment effect.

Once matched, the treated and control groups should be *statistically indistinguishable* in terms of $X$.

#### Mathematical Foundation

Under the Strong Ignorability Assumption:
$$
Y(0), Y(1) \perp T \mid X, \quad 0 < e(X) < 1
$$

the average treatment effect (ATE) can be estimated as:
$$
\text{ATE} = E[Y(1) - Y(0)] = E_T\left[\frac{Y T}{e(X)} - \frac{Y (1 - T)}{1 - e(X)}\right]
$$

After matching, we compute the average treatment effect on the treated (ATT):
$$
\text{ATT} = E[Y(1) - Y(0) \mid T = 1]
$$

where $Y(0)$ for treated units is replaced by outcomes of matched control units.

#### Tiny Code (Propensity Score Matching)

Python example using logistic regression and nearest-neighbor matching

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

def propensity_score_matching(X, T, Y):
    # Step 1: estimate propensity scores
    model = LogisticRegression()
    model.fit(X, T)
    e = model.predict_proba(X)[:, 1]

    # Step 2: match treated to control by nearest propensity
    treated_idx = np.where(T == 1)[0]
    control_idx = np.where(T == 0)[0]

    nbrs = NearestNeighbors(n_neighbors=1).fit(e[control_idx].reshape(-1, 1))
    _, match_idx = nbrs.kneighbors(e[treated_idx].reshape(-1, 1))
    matched_control = control_idx[match_idx.flatten()]

    # Step 3: compute ATT
    att = np.mean(Y[treated_idx] - Y[matched_control])
    return att, e
```

#### Why It Matters

- Simulates randomization when true experiments are impossible.
- Reduces selection bias by balancing covariates.
- Works well in economics, medicine, and social science.
- Provides interpretable causal estimates.

But note: it cannot fix *hidden confounding*, only observed variables are adjusted.

#### Try It Yourself

1. Use a dataset with treatment $T$, outcome $Y$, and covariates $X$.
2. Estimate propensity scores using logistic regression.
3. Match each treated unit to one control with similar $e(x)$.
4. Compute the mean difference in outcomes.
5. Check covariate balance before and after matching.

#### Test Cases

| Dataset          | Method | Estimated ATT | Comment                        |
| ---------------- | ------ | ------------- | ------------------------------ |
| Synthetic Linear | PSM    | +2.1          | True effect = +2               |
| Lalonde          | PSM    | +1.6          | Matches experimental benchmark |
| IHDP             | PSM    | +3.3          | Close to true causal value     |

#### Complexity

| Step             | Time          | Space  |
| ---------------- | ------------- | ------ |
| Propensity model | $O(n d)$      | $O(d)$ |
| Matching         | $O(n \log n)$ | $O(n)$ |
| ATT computation  | $O(n)$        | $O(1)$ |

where $d$ = number of covariates.
Overall, efficient and scalable for moderate datasets.

#### A Gentle Proof (Why It Works)

The propensity score is a balancing score, given $e(X)$, treatment assignment is independent of covariates:
$$
T \perp X \mid e(X)
$$
Therefore, matching on $e(X)$ ensures the treated and control groups are comparable, just as in randomized experiments.

By conditioning on $e(X)$ instead of full $X$, PSM compresses high-dimensional adjustment into a single scalar, the probability of treatment.

#### Summary Table

| Concept        | Description                         |
| -------------- | ----------------------------------- |
| Framework      | Causal Inference                    |
| Key Assumption | No unobserved confounders           |
| Core Step      | Match on $P(T=1 \mid X)$            |
| Output         | Average Treatment Effect (ATE/ATT)  |
| Benefit        | Mimics randomized control           |
| Limitation     | Sensitive to model misspecification |

Propensity Score Matching bridges the gap between *data we have* and *experiments we wish we'd run*, balancing worlds to reveal cause beneath correlation.

### 987. Instrumental Variable Estimation

Instrumental Variable (IV) Estimation is a classic method in causal inference used when there are unobserved confounders, variables that affect both the treatment and the outcome, making ordinary regression biased.
It introduces an external variable, called an instrument, that influences the treatment but has *no direct effect* on the outcome except through that treatment.

This approach is essential in econometrics, epidemiology, and policy analysis when randomization is impossible.

#### What Problem Are We Solving?

Suppose we want to estimate the causal effect of treatment $T$ on outcome $Y$.
A simple linear model might look like:
$$
Y = \beta T + \gamma X + \epsilon
$$

If $T$ is correlated with the error term $\epsilon$ (due to hidden confounders), then ordinary least squares (OLS) gives a biased estimate of $\beta$.

We need an external source of variation in $T$ that is not related to $\epsilon$.

#### The Core Idea

We find an instrumental variable $Z$ satisfying:

1. Relevance: $Z$ is correlated with treatment $T$
   $$\text{Cov}(Z, T) \neq 0$$
2. Exogeneity: $Z$ affects $Y$ only through $T$
   $$Z \perp \epsilon$$

Then, $Z$ serves as a *natural experiment*, providing variation in $T$ that is uncorrelated with the confounders.

The key causal relationship becomes:
$$
Z \to T \to Y
$$
and no direct edge from $Z$ to $Y$.

#### Two-Stage Least Squares (2SLS)

The standard procedure for IV estimation is Two-Stage Least Squares (2SLS).

| Stage       | Description                                                                                                |
| ----------- | ---------------------------------------------------------------------------------------------------------- |
| Stage 1 | Regress $T$ on $Z$ and covariates $X$:  $$T = \pi_0 + \pi_1 Z + \pi_2 X + \nu$$                            |
| Stage 2 | Regress $Y$ on the predicted $\hat{T}$ from Stage 1:  $$Y = \beta_0 + \beta_1 \hat{T} + \beta_2 X + \eta$$ |

The coefficient $\beta_1$ is the causal effect of $T$ on $Y$.

Intuitively, $\hat{T}$ is the "clean" part of $T$ explained by $Z$, stripped of confounding influence.

#### Mathematical Derivation

In the simple case without covariates:
$$
Y = \beta T + \epsilon
$$
and $Z$ is the instrument.

The IV estimator is given by:
$$
\hat{\beta}_{IV} = \frac{\text{Cov}(Z, Y)}{\text{Cov}(Z, T)}
$$

This ratio measures how much $Y$ changes for each change in $T$ induced by the instrument $Z$.

#### Tiny Code (2SLS Example)

Python example using NumPy

```python
import numpy as np

def iv_estimate(Y, T, Z):
    # Stage 1: Predict treatment using instrument
    Z = np.column_stack((np.ones(len(Z)), Z))
    beta_stage1 = np.linalg.inv(Z.T @ Z) @ Z.T @ T
    T_hat = Z @ beta_stage1

    # Stage 2: Predict outcome using predicted treatment
    X = np.column_stack((np.ones(len(T_hat)), T_hat))
    beta_stage2 = np.linalg.inv(X.T @ X) @ X.T @ Y
    return beta_stage2[1]  # causal coefficient
```

Example use case:

```python
# Simulate data
np.random.seed(0)
Z = np.random.randn(1000)
T = 2*Z + np.random.randn(1000)
Y = 3*T + 0.5*np.random.randn(1000)

beta_iv = iv_estimate(Y, T, Z)
print(beta_iv)  # ≈ 3 (true causal effect)
```

#### Why It Matters

- Solves endogeneity problems (when treatment correlates with noise).
- Enables causal inference without full randomization.
- Forms the basis of natural experiments, e.g.:

  * Using draft lotteries as instruments for military service.
  * Using distance to college as an instrument for education.
  * Using rainfall as an instrument for agricultural production.

#### Try It Yourself

1. Choose a dataset where treatment $T$ is endogenous (correlated with confounders).
2. Identify an external variable $Z$ that affects $T$ but not $Y$ directly.
3. Run both OLS and IV (2SLS) regressions.
4. Compare estimates, IV should correct for bias.
5. Check instrument strength (e.g., first-stage F-statistic > 10).

#### Test Cases

| Scenario              | Instrument          | True Effect | OLS Estimate | IV Estimate |
| --------------------- | ------------------- | ----------- | ------------ | ----------- |
| Education → Wage      | Distance to college | +0.10       | +0.18        | +0.11       |
| Smoking → Birthweight | Cigarette tax       | -200        | -80          | -190        |
| Alcohol → Accidents   | Beer tax            | +1.5        | +0.7         | +1.4        |

#### Complexity

| Step               | Time       | Space    |
| ------------------ | ---------- | -------- |
| Stage 1 regression | $O(n d^2)$ | $O(d^2)$ |
| Stage 2 regression | $O(n d^2)$ | $O(d^2)$ |
| Total              | $O(n d^2)$ | $O(d)$   |

Efficient and scalable for moderate-dimensional linear models.

#### A Gentle Proof (Why It Works)

Since $Z$ is independent of $\epsilon$, we can take expectations:
$$
E[Z Y] = E[Z (\beta T + \epsilon)] = \beta E[Z T]
$$
thus:
$$
\beta = \frac{E[Z Y]}{E[Z T]}
$$
which proves that $\hat{\beta}_{IV}$ consistently estimates the true causal effect.

This holds even when $T$ is correlated with $\epsilon$, as long as $Z$ is not.

#### Summary Table

| Concept       | Description                                                    |
| ------------- | -------------------------------------------------------------- |
| Framework     | Causal Inference (Endogeneity)                                 |
| Key Idea      | Use exogenous variable $Z$ as a proxy for randomization        |
| Core Equation | $\hat{\beta}_{IV} = \frac{\text{Cov}(Z, Y)}{\text{Cov}(Z, T)}$ |
| Algorithm     | Two-Stage Least Squares (2SLS)                                 |
| Strength      | Handles unobserved confounding                                 |
| Limitation    | Requires strong, valid instrument                              |

Instrumental Variable Estimation is the art of finding natural randomness in the world, turning everyday variation into experiments nature already ran for us.

### 988. Robust Optimization

Robust Optimization (RO) is a framework for making decisions that remain effective under uncertainty.
Instead of assuming perfect knowledge of parameters, RO prepares for *the worst plausible case*, ensuring that a solution performs well even when reality deviates from the model.

It is widely used in finance, operations research, and machine learning to guard against estimation errors, data noise, and adversarial uncertainty.

#### What Problem Are We Solving?

Many optimization problems assume exact parameters, but in practice, coefficients, costs, or probabilities may be uncertain.
For example:

- In portfolio optimization, expected returns are estimated from noisy data.
- In supply chains, demand and cost forecasts fluctuate daily.
- In machine learning, loss surfaces can shift across domains.

Traditional optimization may fail catastrophically when inputs differ from expectations.
Robust optimization explicitly models uncertainty and protects solutions against it.

#### The Core Idea

Let's start with a standard linear optimization problem:

$$
\min_x ; c^T x \quad \text{s.t.} \quad A x \le b
$$

In robust optimization, we treat coefficients as uncertain within specified *uncertainty sets*.
For example, each coefficient $a_{ij}$ may vary within a known range:
$$
a_{ij} \in [\bar{a}*{ij} - \Delta*{ij}, ; \bar{a}*{ij} + \Delta*{ij}]
$$

Then we require the constraint to hold for all possible realizations:
$$
A x \le b, \quad \forall A \in \mathcal{U}
$$

where $\mathcal{U}$ is the uncertainty set (e.g., intervals, ellipsoids, or polyhedra).

This converts the problem into a min–max form:
$$
\min_x ; \max_{u \in \mathcal{U}} f(x, u)
$$

We seek a solution $x^*$ that minimizes the *worst-case loss*.

#### Types of Uncertainty Sets

| Type        | Example                                   | Mathematical Form                                              |                 |                |
| ----------- | ----------------------------------------- | -------------------------------------------------------------- | --------------- | -------------- |
| Box         | Each parameter varies within fixed bounds | $\mathcal{U} = {a:                                             | a_i - \bar{a}_i | \le \Delta_i}$ |
| Ellipsoidal | Uncertainty correlated across parameters  | $\mathcal{U} = {a: (a - \bar{a})^T Q^{-1}(a - \bar{a}) \le 1}$ |                 |                |
| Polyhedral  | Constraints define feasible region        | $\mathcal{U} = {a: F a \le g}$                                 |                 |                |

Each choice yields a different computational complexity and conservativeness.

#### Example: Robust Linear Program

Nominal problem:
$$
\min_x ; c^T x \quad \text{s.t.} \quad A x \le b
$$

Robust version (box uncertainty):
$$
A = \bar{A} + \Delta, \quad |\Delta_{ij}| \le \rho_{ij}
$$

Then the robust constraint becomes:
$$
\bar{a}*i^T x + \sum_j \rho*{ij} |x_j| \le b_i
$$

This yields a convex reformulation that can be solved using standard LP or conic solvers.

#### Tiny Code (Robust LP Example)

Python example using `cvxpy`

```python
import cvxpy as cp
import numpy as np

# Nominal data
A = np.array([[2, 1], [1, 3]])
b = np.array([5, 7])
c = np.array([1, 2])
rho = np.array([[0.1, 0.2], [0.3, 0.1]])

# Decision variable
x = cp.Variable(2, nonneg=True)

# Robust constraints
constraints = []
for i in range(A.shape[0]):
    constraints.append(A[i] @ x + np.sum(rho[i] * cp.abs(x)) <= b[i])

# Objective
objective = cp.Minimize(c @ x)
problem = cp.Problem(objective, constraints)
problem.solve()

print("Robust solution:", x.value)
```

This ensures constraints hold even when coefficients deviate by ±ρ.

#### Why It Matters

- Guarantees feasibility under uncertainty.
- Prevents overfitting to a single estimate.
- Useful in adversarial ML, finance, logistics, and resource allocation.
- Generalizes to robust regression, robust SVM, and robust neural training.

In machine learning, for instance, robust optimization underpins adversarial training:
$$
\min_\theta \max_{\delta \in \mathcal{U}} L(f_\theta(x + \delta), y)
$$
which ensures the model performs well even under worst-case perturbations.

#### Try It Yourself

1. Create a simple LP model $\min c^T x$ subject to $A x \le b$.
2. Add box uncertainty on $A$.
3. Compare the nominal and robust solutions.
4. Increase uncertainty magnitude $\rho$, observe how $x^*$ becomes more conservative.

#### Test Cases

| Scenario               | Uncertainty Set | Robust vs. Nominal Cost              | Notes                              |
| ---------------------- | --------------- | ------------------------------------ | ---------------------------------- |
| Portfolio optimization | Ellipsoidal     | Slightly higher risk-adjusted return | Less sensitive to estimation noise |
| Supply chain           | Box             | Higher cost, fewer stockouts         | Reliable under demand uncertainty  |
| Scheduling             | Polyhedral      | Stable schedule                      | Resistant to delay variations      |

#### Complexity

| Step         | Time     | Space    |
| ------------ | -------- | -------- |
| LP (nominal) | $O(n^3)$ | $O(n^2)$ |
| Robust LP    | $O(n^3)$ | $O(n^2)$ |
| Robust SOCP  | $O(n^3)$ | $O(n^2)$ |

The complexity remains polynomial when $\mathcal{U}$ is convex (box or ellipsoidal).

#### A Gentle Proof (Why It Works)

Let the uncertain constraint be:
$$
a^T x \le b, \quad \forall a \in \mathcal{U}
$$
Then for a convex uncertainty set $\mathcal{U}$, the *worst-case* $a$ occurs at the boundary of $\mathcal{U}$,
turning the infinite constraints into finite deterministic constraints.

For box uncertainty:
$$
\max_{|a_i - \bar{a}_i| \le \rho_i} a^T x = \bar{a}^T x + \sum_i \rho_i |x_i|
$$
This converts the problem into a convex one, a robust version of the nominal LP.

#### Summary Table

| Concept      | Description                               |
| ------------ | ----------------------------------------- |
| Framework    | Optimization under uncertainty            |
| Key Idea     | Protect against worst-case input          |
| Typical Form | $\min_x \max_{u \in \mathcal{U}} f(x, u)$ |
| Methods      | Box, Ellipsoidal, Polyhedral RO           |
| Benefit      | Reliable, conservative solutions          |
| Limitation   | May be overly cautious                    |

Robust Optimization transforms uncertainty from a threat into a design principle, a way to plan for chaos, not fear it.

### 989. Distributionally Robust Optimization

Distributionally Robust Optimization (DRO) is a powerful generalization of robust optimization that prepares not just for worst-case *parameters*, but for worst-case *distributions* of uncertainty.
Instead of assuming that data come from a known probability distribution, DRO assumes the true distribution may lie within an *ambiguity set* around a nominal estimate, and seeks decisions that perform best under the worst plausible distribution.

This bridges classical stochastic optimization and adversarial learning, and has deep ties to modern machine learning, fairness, and generalization.

#### What Problem Are We Solving?

In many real-world tasks, we train or optimize under data drawn from an unknown distribution $P$, but we only have a finite empirical sample $\hat{P}*n$.
A standard stochastic optimization problem is:
$$
\min*{x} ; E_P[L(x, \xi)]
$$

But $\hat{P}_n$ is just an estimate of $P$. If the training data are biased, incomplete, or nonstationary, the solution may perform poorly on future data.

DRO addresses this by considering all distributions *close* to $\hat{P}*n$, and minimizing the worst-case expected loss:
$$
\min*{x} ; \max_{Q \in \mathcal{P}} E_Q[L(x, \xi)]
$$

where $\mathcal{P}$ is an ambiguity set of plausible distributions.

#### The Core Idea

The ambiguity set $\mathcal{P}$ defines how much deviation from the nominal distribution is allowed. Common choices include:

| Type                  | Definition                                                           | Notes                                      |
| --------------------- | -------------------------------------------------------------------- | ------------------------------------------ |
| $\phi$-divergence | $\mathcal{P} = { Q : D_\phi(Q \Vert \hat{P}) \le \rho }$             | Includes KL, $\chi^2$, and total variation |
| Wasserstein ball  | $\mathcal{P} = { Q : W(Q, \hat{P}) \le \epsilon }$                   | Distance in probability metric space       |
| Moment-based      | $\mathcal{P} = { Q : E_Q[\xi] = \mu, \text{Var}_Q[\xi] \le \Sigma }$ | Controls moments rather than shape         |

Then, DRO finds a decision $x^*$ that minimizes *expected loss* under the worst distribution $Q$ within $\mathcal{P}$.

#### Mathematical Form

Given a loss function $L(x, \xi)$ and ambiguity set $\mathcal{P}$:
$$
\min_x ; \max_{Q \in \mathcal{P}} E_Q[L(x, \xi)]
$$

The inner maximization over $Q$ represents an adversarial "nature" choosing the worst-case distribution within $\mathcal{P}$.

Under mild convexity assumptions, this problem is dually equivalent to a regularized empirical risk minimization problem:
$$
\min_x ; E_{\hat{P}}[L(x, \xi)] + \lambda , \Omega(x)
$$
where $\Omega(x)$ is a regularizer depending on the ambiguity type.

Thus, DRO unifies robustness and regularization, a core insight in modern ML.

#### Example: DRO with Wasserstein Distance

The Wasserstein ambiguity set is defined as:
$$
\mathcal{P} = { Q : W(Q, \hat{P}_n) \le \epsilon }
$$
where $W$ is the optimal transport (Earth mover's) distance.

The DRO objective becomes:
$$
\min_x \max_{W(Q, \hat{P}_n) \le \epsilon} E_Q[L(x, \xi)]
$$

This has a dual formulation:
$$
\min_x ; E_{\hat{P}*n}[L(x, \xi)] + \epsilon \cdot | \nabla_x L(x, \xi) |
$$
which resembles *adversarial training* in neural networks.

#### Tiny Code (Wasserstein DRO Example)

Python example using convex loss

```python
import numpy as np

def wasserstein_dro_loss(L, grad_L, x, epsilon):
    """Approximate DRO regularized loss."""
    empirical_loss = np.mean(L(x))
    grad_norm = np.linalg.norm(np.mean(grad_L(x)), ord=2)
    return empirical_loss + epsilon * grad_norm
```

This simple structure is analogous to:
$$
\text{Loss}_{DRO} = \text{EmpiricalLoss} + \epsilon \times \text{GradientNorm}
$$
— a robustified loss against distributional shifts.

#### Why It Matters

- Provides robust generalization beyond training data.
- Protects against data shift, bias, and sampling noise.
- Bridges optimization and statistical learning theory.
- Interpretable as adversarial regularization in deep learning.

In machine learning, DRO yields models that perform consistently across different domains, crucial for fairness, out-of-distribution generalization, and reliable AI.

#### Try It Yourself

1. Train a linear regression model on a biased dataset.
2. Compute a Wasserstein-based DRO objective.
3. Tune $\epsilon$, observe that small $\epsilon$ improves robustness, large $\epsilon$ yields overly conservative solutions.
4. Compare test performance under domain shift.

#### Test Cases

| Task                   | Ambiguity Set | Effect                 | Outcome                           |
| ---------------------- | ------------- | ---------------------- | --------------------------------- |
| Logistic regression    | KL-divergence | Regularizes weights    | Improved generalization           |
| Portfolio optimization | Wasserstein   | Hedge against shocks   | Reduced volatility                |
| Image classification   | Wasserstein   | Adversarial robustness | Stronger against FGSM/PGD attacks |

#### Complexity

| Step                              | Time              | Space    |
| --------------------------------- | ----------------- | -------- |
| Inner distributional optimization | $O(n)$ – $O(n^2)$ | $O(n)$   |
| Dual formulation                  | $O(d^3)$          | $O(d^2)$ |
| Gradient-based updates            | $O(n d)$          | $O(d)$   |

Modern DRO solvers leverage convex duality or stochastic gradient methods for scalability.

#### A Gentle Proof (Why It Works)

By strong duality, for many $\mathcal{P}$, the inner supremum admits a closed-form expression:
$$
\max_{Q \in \mathcal{P}} E_Q[L(x, \xi)] = E_{\hat{P}_n}[L(x, \xi)] + \text{robust penalty}
$$
where the penalty corresponds to the *worst perturbation* consistent with $\mathcal{P}$.

This means DRO implicitly regularizes the solution to remain stable under data perturbations, connecting to Tikhonov and adversarial regularization.

#### Summary Table

| Concept      | Description                                           |
| ------------ | ----------------------------------------------------- |
| Framework    | Optimization under distributional uncertainty         |
| Core Idea    | Minimize loss under worst plausible data distribution |
| Key Form     | $\min_x \max_{Q \in \mathcal{P}} E_Q[L(x, \xi)]$      |
| Common Sets  | Wasserstein, KL, $\chi^2$, Moment-based               |
| Dual Form    | Regularized empirical risk                            |
| Applications | ML robustness, fairness, finance                      |
| Limitation   | Can be overly conservative if $\mathcal{P}$ too large |

Distributionally Robust Optimization is the mathematical heart of trustworthy AI, ensuring our models don't just fit the data they see, but also withstand the worlds they haven't yet encountered.

### 990. Counterfactual Fairness

Counterfactual Fairness is a fairness criterion grounded in causal reasoning.
It asks: *Would this decision have been the same if the individual had belonged to a different demographic group, all else being equal?*

Rather than relying on correlations between protected attributes (like race or gender) and outcomes, counterfactual fairness uses causal models to reason about how changing sensitive attributes would affect predictions.

#### What Problem Are We Solving?

Machine learning models often encode biases from historical data. Even if sensitive attributes are excluded, proxies or correlated variables can still leak bias.

For example:

- A credit scoring model may discriminate based on postal codes correlated with race.
- A hiring model may favor certain universities that correlate with socioeconomic status.

We want models that make the same decision for the same person, even if we *hypothetically change* their protected attribute, that is, under a *counterfactual world*.

#### The Core Idea

Let:

- $A$: protected attribute (e.g., gender, race)
- $X$: observed features (e.g., education, income)
- $Y$: outcome (e.g., loan approval)
- $\hat{Y}$: model prediction

A model is counterfactually fair if:
$$
P(\hat{Y}*{A \leftarrow a}(U) = y \mid X = x, A = a) = P(\hat{Y}*{A \leftarrow a'}(U) = y \mid X = x, A = a)
$$
for all possible values $a, a'$ of the protected attribute.

Here, $\hat{Y}_{A \leftarrow a}(U)$ represents the *counterfactual prediction* if we were to intervene and set $A = a$ in the causal model, keeping all other latent factors $U$ fixed.

Intuitively:

> A model is fair if the prediction for an individual would not change in a hypothetical world where only their sensitive attribute were different.

#### Causal Graph Perspective

The causal structure is modeled using a Structural Causal Model (SCM):

$$
A \to X \to \hat{Y}, \quad A \to \hat{Y}
$$

Counterfactual fairness involves removing or blocking the *unfair causal paths* from $A$ to $\hat{Y}$ that do not pass through legitimate mediators.

For example:

- Path $A \to X \to \hat{Y}$ (e.g., education) may be acceptable.
- Direct path $A \to \hat{Y}$ (e.g., race influencing prediction) is *unfair*.

#### Example Scenario

Hiring decision model
Variables:

- $A$: gender
- $X$: years of experience, education
- $\hat{Y}$: predicted suitability

If gender affects education (due to structural inequality) but not intrinsic ability, we may wish to remove the direct effect $A \to \hat{Y}$ but keep the mediated path $A \to X \to \hat{Y}$.

We simulate the counterfactual:

- Keep latent ability $U$ constant.
- Change $A$ from male to female.
- Recompute $\hat{Y}$ under the modified graph.

If $\hat{Y}$ changes, unfairness exists.

#### Tiny Code (Simplified Counterfactual Simulation)

Python pseudocode using DoWhy

```python
import dowhy
from dowhy import CausalModel

# Example data with bias
data = {
    "A": [0, 1, 0, 1, 0],        # gender
    "X": [10, 12, 9, 11, 10],    # experience
    "Y": [1, 0, 1, 0, 1]         # hiring decision
}

model = CausalModel(
    data=data,
    treatment="A",
    outcome="Y",
    common_causes=["X"]
)

identified_estimand = model.identify_effect()
estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")

print("Estimated causal effect of gender:", estimate.value)
```

You can then perform a do-intervention (`do(A=0)` vs. `do(A=1)`) to test whether $\hat{Y}$ changes under the counterfactual scenario.

#### Why It Matters

- Ensures fairness grounded in causality, not mere correlation.
- Detects hidden discrimination that persists even after removing $A$ from features.
- Encourages interpretable fairness definitions aligned with legal and ethical reasoning.

Applications include:

- Credit risk and loan approvals
- Hiring and promotion models
- Recidivism risk assessment
- Healthcare prioritization

#### Try It Yourself

1. Build a causal graph of your dataset (e.g., using domain knowledge).
2. Identify paths from $A$ to $\hat{Y}$.
3. Remove or neutralize unfair paths.
4. Test whether $\hat{Y}$ changes when you intervene on $A$.
5. Train a model using fair representations or invariant features.

#### Test Cases

| Scenario   | Protected Attribute | Technique                  | Result                        |
| ---------- | ------------------- | -------------------------- | ----------------------------- |
| Hiring     | Gender              | Path blocking (A → Y)      | Equalized predictions         |
| Lending    | Race                | Counterfactual reweighting | Reduced bias                  |
| Healthcare | Age                 | Causal adjustment          | Stable outcomes across groups |

#### Complexity

| Step                      | Time     | Space    |
| ------------------------- | -------- | -------- |
| Causal graph construction | $O(n^2)$ | $O(n^2)$ |
| Intervention computation  | $O(n)$   | $O(n)$   |
| Counterfactual simulation | $O(m n)$ | $O(n)$   |

Most modern frameworks (like DoWhy, EconML, or CausalML) automate these steps efficiently.

#### A Gentle Proof (Why It Works)

Under a structural causal model:
$$
\hat{Y} = f(A, X, U)
$$
where $U$ are latent background variables independent of $A$.

If $\hat{Y}$ is counterfactually fair, then:
$$
f(a, X_{A \leftarrow a}, U) = f(a', X_{A \leftarrow a'}, U)
$$
for all $a, a'$ and fixed $U$.

This implies $\hat{Y}$ depends only on *legitimate features* and not on $A$ or its unfair descendants.

#### Summary Table

| Concept          | Description                                                                                  |
| ---------------- | -------------------------------------------------------------------------------------------- |
| Framework        | Causal fairness in decision systems                                                          |
| Key Idea         | Prediction invariant under counterfactual change in sensitive attribute                      |
| Formal Criterion | $P(\hat{Y}*{A \leftarrow a} = y \mid X, A=a) = P(\hat{Y}*{A \leftarrow a'} = y \mid X, A=a)$ |
| Implementation   | Structural Causal Models + do-calculus                                                       |
| Benefit          | Detects and corrects hidden bias                                                             |
| Limitation       | Requires causal knowledge and assumptions                                                    |

Counterfactual Fairness is the gold standard of algorithmic fairness, it asks not whether two groups are treated equally *on average*, but whether each *individual* would be treated the same in every possible world.

# Section 100. AI Planning, Search and Learning Systems 

### 991. Breadth-First Search (BFS)

Breadth-First Search (BFS) is one of the most fundamental graph traversal algorithms.
It explores a graph level by level, discovering all vertices at distance 1 before distance 2, and so on.
This makes BFS ideal for finding the shortest path in unweighted graphs or performing systematic exploration of networks, trees, or state spaces.

#### What Problem Are We Solving?

We want to traverse or search through a graph, visiting all reachable vertices from a given start node.

Given a graph $G = (V, E)$ and a starting node $s$, BFS finds:

- All nodes reachable from $s$.
- The shortest distance $d(s, v)$ for each vertex $v$.
- The parent or predecessor relationships (to reconstruct paths).

Formally, BFS explores vertices in increasing order of distance from the source.

#### The Core Idea

1. Maintain a queue of vertices to explore.
2. Start with the source node and mark it as visited.
3. Repeatedly dequeue a node, visit all its unvisited neighbors, and enqueue them.
4. Continue until the queue is empty.

The algorithm guarantees that each vertex is visited once, and edges are relaxed in breadth-first order.

#### Algorithm

| Step | Description                                                                  |
| ---- | ---------------------------------------------------------------------------- |
| 1    | Initialize queue $Q$ with start node $s$.                                    |
| 2    | Mark $s$ as visited and set $d(s) = 0$.                                      |
| 3    | While $Q$ not empty:                                                         |
|      |   a. Pop vertex $u$ from $Q$.                                                |
|      |   b. For each neighbor $v$ of $u$:                                           |
|      |     i. If $v$ not visited: mark visited, set $d(v) = d(u) + 1$, enqueue $v$. |

#### Tiny Code (Classic BFS)

C Example

```c
#include <stdio.h>
#include <stdbool.h>

#define N 100
int adj[N][N];   // adjacency matrix
bool visited[N];
int queue[N];
int front = 0, rear = 0;

void enqueue(int v) { queue[rear++] = v; }
int dequeue() { return queue[front++]; }

void bfs(int start, int n) {
    for (int i = 0; i < n; i++) visited[i] = false;

    visited[start] = true;
    enqueue(start);

    while (front < rear) {
        int u = dequeue();
        printf("%d ", u);
        for (int v = 0; v < n; v++) {
            if (adj[u][v] && !visited[v]) {
                visited[v] = true;
                enqueue(v);
            }
        }
    }
}
```

Python Example

```python
from collections import deque

def bfs(graph, start):
    visited = set([start])
    queue = deque([start])
    order = []

    while queue:
        u = queue.popleft()
        order.append(u)
        for v in graph[u]:
            if v not in visited:
                visited.add(v)
                queue.append(v)
    return order
```

Example usage:

```python
G = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

print(bfs(G, 'A'))  # ['A', 'B', 'C', 'D', 'E', 'F']
```

#### Why It Matters

- Shortest paths in unweighted graphs.
- Connectivity checks in graphs and networks.
- Component detection in undirected graphs.
- Layered search in AI planning or puzzles.

BFS is the foundation for many algorithms:

- Dijkstra (with weights)
- Ford–Fulkerson (augmenting paths)
- Edmonds–Karp (max flow)
- Topological order detection in DAGs

#### Try It Yourself

1. Implement BFS using an adjacency list and verify traversal order.
2. Compute distance of every vertex from the source.
3. Use BFS to find the shortest path in an unweighted maze.
4. Modify BFS to count connected components in an undirected graph.

#### Test Cases

| Graph                       | Start | Expected Order |
| --------------------------- | ----- | -------------- |
| A–B–C–D                     | A     | A, B, C, D     |
| Triangle A–B–C–A            | B     | B, A, C        |
| Star (1 connected to 2,3,4) | 1     | 1, 2, 3, 4     |

#### Complexity

| Measure | Complexity                     |
| ------- | ------------------------------ |
| Time    | $O(V + E)$                     |
| Space   | $O(V)$ (queue + visited array) |

Each vertex and edge is processed once, giving linear time in graph size.

#### A Gentle Proof (Why It Works)

At every iteration, BFS dequeues the vertex with minimum distance from the source.
Since it visits neighbors in order of increasing distance, the first time a vertex is discovered corresponds to the shortest path from the source.

Formally, by induction:

- Base: $d(s) = 0$.
- Step: if $u$ has correct distance $d(u)$, all unvisited neighbors get $d(u) + 1$, preserving order.

Hence, BFS produces correct shortest distances in unweighted graphs.

#### Summary Table

| Concept          | Description                          |
| ---------------- | ------------------------------------ |
| Framework        | Graph traversal                      |
| Core Idea        | Explore neighbors layer by layer     |
| Data Structure   | Queue (FIFO)                         |
| Guarantee        | Shortest path in unweighted graphs   |
| Time Complexity  | $O(V + E)$                           |
| Space Complexity | $O(V)$                               |
| Applications     | Search, connectivity, shortest paths |

Breadth-First Search is the algorithmic heartbeat of exploration, calm, orderly, and fair: visiting every neighbor before wandering deeper into the graph's unknown.

### 992. Depth-First Search (DFS)

Depth-First Search (DFS) explores a graph by going as deep as possible along one branch before backtracking.
It's the opposite of BFS, instead of expanding breadth-first layers, DFS dives downward through edges, uncovering hidden structures like paths, cycles, and connectivity patterns.

#### What Problem Are We Solving?

Given a graph $G = (V, E)$ and a starting vertex $s$, we want to explore all vertices reachable from $s$.
Unlike BFS, which guarantees shortest paths, DFS focuses on complete exploration, ideal for:

- Detecting connected components
- Checking cycles in graphs
- Generating topological orderings
- Solving maze traversal and pathfinding tasks

#### The Core Idea

DFS uses a stack (explicit or recursive) to remember where it came from.
It works recursively as:

1. Visit a vertex.
2. Recurse into each unvisited neighbor.
3. Backtrack when no unvisited neighbors remain.

This "deep first" pattern ensures that every path is explored to its end before moving on.

#### Algorithm

| Step | Description                                                       |
| ---- | ----------------------------------------------------------------- |
| 1    | Mark the starting node as visited.                                |
| 2    | For each neighbor $v$ of the node:                                |
|      |   a. If $v$ not visited, recurse with DFS($v$).                   |
| 3    | Continue until all vertices reachable from the start are visited. |

#### Tiny Code (Classic DFS)

C Example (Recursive)

```c
#include <stdio.h>
#include <stdbool.h>

#define N 100
int adj[N][N];
bool visited[N];

void dfs(int v, int n) {
    visited[v] = true;
    printf("%d ", v);
    for (int u = 0; u < n; u++) {
        if (adj[v][u] && !visited[u]) {
            dfs(u, n);
        }
    }
}
```

Python Example

```python
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    for v in graph[start]:
        if v not in visited:
            dfs(graph, v, visited)
    return visited
```

Example usage:

```python
G = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

print(dfs(G, 'A'))  # {'A', 'B', 'D', 'E', 'F', 'C'}
```

#### Why It Matters

DFS underlies many advanced algorithms:

- Cycle detection in directed or undirected graphs
- Topological sorting in DAGs
- Finding articulation points and bridges
- Tarjan's SCC algorithm for strongly connected components
- Backtracking in puzzles (Sudoku, N-Queens, pathfinding)

In trees, DFS is the natural traversal for pre-order, in-order, and post-order searches.

#### Try It Yourself

1. Modify DFS to record discovery and finishing times.
2. Use DFS to detect whether a graph is connected.
3. Apply DFS on a directed graph and perform topological sorting.
4. Visualize the DFS recursion tree, mark back edges and forward edges.

#### Test Cases

| Graph              | Start | Expected Order                    |
| ------------------ | ----- | --------------------------------- |
| A–B–C              | A     | A, B, C                           |
| Triangle (A–B–C–A) | B     | B, A, C                           |
| Star (1→2,3,4)     | 1     | 1, 2, 3, 4 (depth order may vary) |

#### Complexity

| Measure | Complexity                        |
| ------- | --------------------------------- |
| Time    | $O(V + E)$                        |
| Space   | $O(V)$ (stack or recursion depth) |

Each vertex and edge is visited once.
Space complexity equals recursion depth, which in the worst case can reach $V$.

#### A Gentle Proof (Why It Works)

DFS explores along a path until no unvisited neighbor remains, then backtracks.
Each edge $(u, v)$ is considered exactly once when $v$ is first discovered.
By induction, DFS guarantees every reachable vertex is visited exactly once.

#### Variants

| Variant       | Description                                     |
| ------------- | ----------------------------------------------- |
| Recursive DFS | Simpler, uses call stack                        |
| Iterative DFS | Explicit stack, avoids recursion limit          |
| Modified DFS  | Tracks parent edges or discovery times          |
| DFS Forest    | Collection of DFS trees for disconnected graphs |

#### Summary Table

| Concept          | Description                                         |
| ---------------- | --------------------------------------------------- |
| Framework        | Graph traversal                                     |
| Core Idea        | Explore deeply before backtracking                  |
| Data Structure   | Stack or recursion                                  |
| Guarantee        | Full exploration of connected component             |
| Time Complexity  | $O(V + E)$                                          |
| Space Complexity | $O(V)$                                              |
| Applications     | Topological sort, SCC, pathfinding, cycle detection |

Depth-First Search is the explorer's mindset, fearless, methodical, and curious, diving deep into one path until it has seen all that lies beneath.

### 993. A* Search

A* (pronounced "A-star") is one of the most influential search algorithms in artificial intelligence.
It combines the cost so far with a heuristic estimate of the cost to go, balancing efficiency and optimality.
A* elegantly generalizes Dijkstra's algorithm and greedy best-first search, and it's used everywhere from game AI to robotics to route planning.

#### What Problem Are We Solving?

We want to find the shortest path between a start node and a goal node in a weighted graph.
Each edge has a cost $c(u, v)$, and we also have a heuristic function $h(v)$ estimating the remaining cost from $v$ to the goal.

If we only used actual cost $g(v)$ → Dijkstra's algorithm.
If we only used heuristic $h(v)$ → Greedy best-first search.
A* combines both:
$$
f(v) = g(v) + h(v)
$$
The algorithm always expands the node with the smallest total estimated cost $f(v)$.

#### The Core Idea

1. Maintain two sets:

   * Open (frontier): nodes to explore
   * Closed: already visited nodes

2. Start from the source node with $g(s) = 0$.

3. While the open set is not empty:

   * Pick node $u$ with smallest $f(u) = g(u) + h(u)$.
   * If $u$ is the goal, reconstruct path and stop.
   * For each neighbor $v$ of $u$:

     * Compute tentative cost $g'(v) = g(u) + c(u, v)$.
     * If $g'(v) < g(v)$, update and record parent.

#### Algorithm

| Step | Description                                                  |
| ---- | ------------------------------------------------------------ |
| 1    | Initialize open set with start node $s$.                     |
| 2    | Set $g(s) = 0$, $f(s) = h(s)$.                               |
| 3    | While open set not empty:                                    |
|      |   a. Choose node $u$ with smallest $f(u)$.                   |
|      |   b. If $u$ is goal → return path.                           |
|      |   c. For each neighbor $v$ of $u$: update $g(v)$ and $f(v)$. |
|      |   d. Move $u$ to closed set.                                 |

#### Tiny Code (Python Example)

```python
import heapq

def a_star(graph, start, goal, h):
    open_set = [(0, start)]
    came_from = {}
    g = {start: 0}
    f = {start: h(start)}

    while open_set:
        _, u = heapq.heappop(open_set)
        if u == goal:
            path = [u]
            while u in came_from:
                u = came_from[u]
                path.append(u)
            return list(reversed(path))

        for v, cost in graph[u]:
            g_new = g[u] + cost
            if v not in g or g_new < g[v]:
                g[v] = g_new
                f[v] = g_new + h(v)
                came_from[v] = u
                heapq.heappush(open_set, (f[v], v))
    return None
```

Example:

```python
graph = {
    'A': [('B', 1), ('C', 3)],
    'B': [('D', 3)],
    'C': [('D', 1)],
    'D': []
}
h = lambda x: {'A': 3, 'B': 2, 'C': 1, 'D': 0}[x]
print(a_star(graph, 'A', 'D', h))  # ['A', 'C', 'D']
```

#### Why It Matters

- Finds optimal path if heuristic $h$ is admissible ($h(v) \le h^*(v)$, true cost).
- Efficiently prunes paths unlikely to lead to the goal.
- Powers algorithms in:

  * Game pathfinding (e.g., grid maps, navigation meshes)
  * Robotics motion planning
  * Route optimization (maps, logistics)
  * Natural language parsing and AI planning

#### Try It Yourself

1. Implement A* on a 2D grid with obstacles.
2. Test with heuristics:

   * Manhattan distance
   * Euclidean distance
   * Zero heuristic (reduces to Dijkstra)
3. Visualize the expansion of the search frontier.
4. Experiment with overestimating heuristics (non-admissible), see when optimality breaks.

#### Test Cases

| Graph             | Heuristic    | Expected Path           |
| ----------------- | ------------ | ----------------------- |
| A–B–C (unit cost) | $h=0$        | A → B → C               |
| Grid (Manhattan)  | Admissible   | Shortest geometric path |
| Random graph      | Overestimate | May skip optimal path   |

#### Complexity

| Measure | Complexity                                 |
| ------- | ------------------------------------------ |
| Time    | $O(E \log V)$ (using priority queue)       |
| Space   | $O(V)$ (stores $g$, $f$, open/closed sets) |

Efficiency depends heavily on heuristic accuracy, a good $h$ drastically reduces explored nodes.

#### A Gentle Proof (Why It Works)

If $h(v)$ is admissible (never overestimates true cost) and consistent ($h(u) \le c(u,v) + h(v)$),
then A* always expands nodes in order of increasing true cost to the goal.

Therefore, the first time the goal node is removed from the open set, the shortest path has been found.

#### Summary Table

| Concept          | Description                        |
| ---------------- | ---------------------------------- |
| Framework        | Informed graph search              |
| Core Idea        | Minimize total cost $f = g + h$    |
| Guarantee        | Optimal if $h$ admissible          |
| Data Structure   | Priority queue (min-heap)          |
| Time Complexity  | $O(E \log V)$                      |
| Space Complexity | $O(V)$                             |
| Applications     | Pathfinding, robotics, AI planning |

A* is the perfect blend of logic and intuition, a search that not only knows where it's been, but also has a sense of where it should go next.

### 994. Iterative Deepening A* (IDA*)

Iterative Deepening A* combines the space efficiency of Depth-First Search with the optimality and heuristic power of A*.
It's particularly useful when the graph or state space is too large for A* to fit in memory, yet we still need optimal solutions guided by heuristics.

#### What Problem Are We Solving?

Standard A* stores all explored nodes in memory, which can grow exponentially.
For large problems (like puzzles or planning tasks), this becomes infeasible.

IDA* solves this by performing iterative depth-first searches, each limited by an increasing cost threshold.
Instead of exploring by depth, we explore by estimated total cost $f(n) = g(n) + h(n)$, repeating the process with larger limits until the goal is reached.

#### The Core Idea

1. Start with an initial threshold equal to $h(start)$.
2. Perform a Depth-First Search, but prune any node where $f(n) > threshold$.
3. If the goal is not found, increase the threshold to the smallest pruned value and repeat.

This process ensures that the first time the goal is reached, it has the minimum possible $f$ value, guaranteeing optimality.

#### Algorithm

| Step | Description                                                                 |
| ---- | --------------------------------------------------------------------------- |
| 1    | Initialize threshold = $h(start)$.                                          |
| 2    | While true: perform depth-limited DFS using $f(n) = g(n) + h(n)$.           |
| 3    | If goal found, return path.                                                 |
| 4    | If no path within limit, increase threshold to next higher $f$ encountered. |

#### Tiny Code (Python Example)

```python
def ida_star(start, goal, h, neighbors):
    def search(path, g, bound):
        node = path[-1]
        f = g + h(node)
        if f > bound:
            return f
        if node == goal:
            return path
        min_bound = float('inf')
        for next_node, cost in neighbors(node):
            if next_node not in path:
                path.append(next_node)
                t = search(path, g + cost, bound)
                if isinstance(t, list):
                    return t
                if t < min_bound:
                    min_bound = t
                path.pop()
        return min_bound

    bound = h(start)
    path = [start]
    while True:
        t = search(path, 0, bound)
        if isinstance(t, list):
            return t
        if t == float('inf'):
            return None
        bound = t
```

Example:

```python
graph = {
    'A': [('B', 1), ('C', 4)],
    'B': [('D', 1)],
    'C': [('D', 1)],
    'D': []
}
def neighbors(n): return graph.get(n, [])
h = lambda x: {'A': 3, 'B': 2, 'C': 1, 'D': 0}[x]
print(ida_star('A', 'D', h, lambda n: graph.get(n, [])))  # ['A', 'B', 'D']
```

#### Why It Matters

- Memory-efficient A*: uses $O(d)$ space (depth only).
- Guarantees optimal solution if heuristic is admissible.
- Used in domains like:

  * 15-puzzle and sliding puzzles
  * Robot motion planning
  * AI game search
  * Large-scale route finding where A* cannot fit in RAM

#### Try It Yourself

1. Use IDA* on an $8 \times 8$ grid with obstacles.
2. Compare memory usage and number of nodes explored vs A*.
3. Experiment with heuristic tightness (e.g., Manhattan vs Euclidean).
4. Implement cutoff visualization, show how thresholds expand per iteration.

#### Test Cases

| Graph             | Heuristic          | Expected Path                |
| ----------------- | ------------------ | ---------------------------- |
| A–B–D (unit cost) | Admissible         | A → B → D                    |
| Branching with C  | $h(C) > h^*(C)$    | Still finds A → B → D        |
| Grid world        | Manhattan distance | Optimal path found gradually |

#### Complexity

| Measure | Complexity                       |
| ------- | -------------------------------- |
| Time    | $O(b^{d})$ worst case (like DFS) |
| Space   | $O(d)$ (depth of recursion)      |

Although it may revisit nodes across iterations, IDA* saves exponential memory compared to A*.

#### A Gentle Proof (Why It Works)

Each iteration explores all paths with $f(n) \le$ current threshold.
Since thresholds grow in increasing order of $f$, the first time the goal's $f$ value is reached, it must be minimal.
Thus, IDA* maintains both completeness and optimality, provided $h$ is admissible.

#### Summary Table

| Concept          | Description                                   |
| ---------------- | --------------------------------------------- |
| Framework        | Heuristic search with limited memory          |
| Core Idea        | Iteratively deepen cost threshold $f = g + h$ |
| Guarantee        | Optimal if heuristic admissible               |
| Data Structure   | Recursive DFS                                 |
| Time Complexity  | $O(b^{d})$                                    |
| Space Complexity | $O(d)$                                        |
| Applications     | Puzzle solving, pathfinding, planning         |

IDA* searches patiently and wisely, like climbing a mountain in steady, careful thresholds, knowing that each step higher brings you closer to the summit with the least effort possible.

### 995. Uniform Cost Search (UCS)

Uniform Cost Search (UCS) is a classic algorithm for finding the least-cost path between nodes when all edge costs are non-negative.
It is essentially Dijkstra's algorithm framed as a search problem, expanding the least-cost node first, ensuring optimality without needing any heuristic.

#### What Problem Are We Solving?

Given a weighted graph $G = (V, E)$ where each edge $(u, v)$ has a non-negative cost $c(u, v)$,
find the path from a start node $s$ to a goal node $g$ that minimizes the total path cost:
$$
C(s, g) = \sum_{(u, v) \in \text{path}} c(u, v)
$$

Unlike BFS (which assumes unit costs), UCS generalizes to arbitrary positive costs.

#### The Core Idea

- Always expand the node with the smallest known cumulative cost $g(n)$.
- Keep track of the best cost found so far for each node.
- Once a goal node is dequeued from the frontier, the cost is guaranteed to be minimal.

#### Algorithm

| Step | Description                                                    |
| ---- | -------------------------------------------------------------- |
| 1    | Initialize priority queue with start node $(s, 0)$             |
| 2    | While queue not empty:                                         |
|      |   a. Pop node $u$ with smallest cost $g(u)$                    |
|      |   b. If $u$ is the goal → return path                          |
|      |   c. For each neighbor $v$:                                    |
|      |     Compute $g'(v) = g(u) + c(u, v)$                           |
|      |     If $v$ not visited or cheaper path found → update and push |

#### Tiny Code (Python Example)

```python
import heapq

def uniform_cost_search(graph, start, goal):
    pq = [(0, start, [])]
    visited = {}
    while pq:
        cost, node, path = heapq.heappop(pq)
        if node == goal:
            return path + [node], cost
        if node in visited and visited[node] <= cost:
            continue
        visited[node] = cost
        for neighbor, edge_cost in graph[node]:
            heapq.heappush(pq, (cost + edge_cost, neighbor, path + [node]))
    return None, float('inf')
```

Example:

```python
graph = {
    'A': [('B', 1), ('C', 5)],
    'B': [('C', 2), ('D', 4)],
    'C': [('D', 1)],
    'D': []
}
path, cost = uniform_cost_search(graph, 'A', 'D')
print(path, cost)  # ['A', 'B', 'C', 'D'], 4
```

#### Why It Matters

- Finds optimal paths in weighted graphs without requiring heuristics.
- Forms the foundation of A* (where heuristics are added).
- Useful in domains such as:

  * Navigation and logistics
  * Network routing
  * Planning and scheduling
  * AI pathfinding when heuristic not available

#### Try It Yourself

1. Modify UCS to stop after a specific depth.
2. Compare UCS with Dijkstra's algorithm, note the identical frontier logic.
3. Use UCS in a grid world where different terrains have different traversal costs.
4. Visualize how UCS expands nodes in increasing cost order.

#### Test Cases

| Graph                 | Start | Goal  | Expected Path      | Cost   |
| --------------------- | ----- | ----- | ------------------ | ------ |
| A–B–C (1, 2)          | A     | C     | A → B → C          | 3      |
| A–C (5), A–B–C (1, 2) | A     | C     | A → B → C          | 3      |
| Weighted grid         | (0,0) | (2,2) | Minimal-cost route | varies |

#### Complexity

| Measure | Complexity    |
| ------- | ------------- |
| Time    | $O(E \log V)$ |
| Space   | $O(V)$        |

UCS explores nodes in increasing cost order, similar to Dijkstra.
In dense graphs, memory usage can grow as it keeps multiple paths to nodes before pruning.

#### A Gentle Proof (Why It Works)

UCS is optimal because it expands nodes in order of non-decreasing path cost $g(n)$.
When a goal node $g$ is selected for expansion, no cheaper path to $g$ can exist in the queue.
This follows from the monotonicity of edge costs ($c(u, v) \ge 0$).

#### Summary Table

| Concept          | Description                             |
| ---------------- | --------------------------------------- |
| Framework        | Uninformed cost-based search            |
| Core Idea        | Expand node with lowest cumulative cost |
| Guarantee        | Optimal if edge costs non-negative      |
| Data Structure   | Priority queue (min-heap)               |
| Time Complexity  | $O(E \log V)$                           |
| Space Complexity | $O(V)$                                  |
| Applications     | Pathfinding, routing, planning          |

Uniform Cost Search is steady and patient, it never guesses, never rushes.
It simply follows the path of least resistance, one minimal cost at a time, until it reaches the goal with certainty.

### 996. Monte Carlo Tree Search (MCTS)

Monte Carlo Tree Search (MCTS) is the algorithmic heart of modern game-playing AI.
It combines random sampling, incremental tree building, and statistical reasoning to make near-optimal decisions without exhaustive search.
MCTS was famously used in AlphaGo, AlphaZero, and other intelligent agents that learn by exploring possibilities through simulation.

#### What Problem Are We Solving?

We want to find the best move or action in a large search space (like in Go, Chess, or planning tasks)
where the full game tree is too vast to explore completely.

MCTS avoids full expansion by repeatedly simulating random playouts to estimate the value of actions.

Each iteration involves four steps: Selection, Expansion, Simulation, Backpropagation.

#### The Core Idea

Each node in the tree represents a state.
The algorithm grows the tree asymmetrically, focusing more on promising branches.

At each iteration:

1. Selection:
   Traverse the tree from root using a *tree policy* (like UCT) until a node is not fully expanded.
   Select child $i$ that maximizes:
   $$
   UCT(i) = \bar{X}_i + c \sqrt{\frac{\ln N}{n_i}}
   $$
   where $\bar{X}_i$ is average reward, $n_i$ visits, and $N$ total visits of parent.

2. Expansion:
   Add one or more unexplored children.

3. Simulation:
   Run a random rollout (playout) from that new node until the game ends.
   Record the outcome (win/loss/score).

4. Backpropagation:
   Update visit counts and rewards along the path back to the root.

Over time, estimates converge, and the root's best child corresponds to the best move.

#### Algorithm Outline

| Step | Description                                       |
| ---- | ------------------------------------------------- |
| 1    | Start from root (current game state).             |
| 2    | Repeat N iterations:                              |
|      | a. Select promising node using UCT.               |
|      | b. Expand by generating a new child.              |
|      | c. Simulate a random game from that node.         |
|      | d. Backpropagate results up the tree.             |
| 3    | Return the child with the highest average reward. |

#### Tiny Code (Python Example)

```python
import math, random

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

def uct(node, c=1.414):
    if node.visits == 0:
        return float('inf')
    return node.value / node.visits + c * math.sqrt(math.log(node.parent.visits) / node.visits)

def mcts(root, iter_limit, get_moves, simulate, make_move):
    for _ in range(iter_limit):
        node = root
        # Selection
        while node.children and all(c.visits > 0 for c in node.children):
            node = max(node.children, key=uct)
        # Expansion
        moves = get_moves(node.state)
        if moves:
            move = random.choice(moves)
            new_state = make_move(node.state, move)
            child = Node(new_state, parent=node)
            node.children.append(child)
            node = child
        # Simulation
        result = simulate(node.state)
        # Backpropagation
        while node:
            node.visits += 1
            node.value += result
            node = node.parent
    return max(root.children, key=lambda n: n.value / n.visits)
```

#### Why It Matters

- Balances exploration (try new moves) and exploitation (favor known good moves).
- Scales well to extremely large search spaces.
- Used in:

  * Game AI (Go, Chess, Hex, Shogi, video games)
  * Planning and robotics
  * Reinforcement learning hybrids (AlphaZero)

#### Try It Yourself

1. Implement MCTS on Tic-Tac-Toe or Connect Four.
2. Change exploration constant $c$ to see trade-off behavior.
3. Compare pure random playouts vs heuristic-guided playouts.
4. Add time limits instead of iteration limits.

#### Test Cases

| Domain          | Action Space | Behavior                   |
| --------------- | ------------ | -------------------------- |
| Tic-Tac-Toe     | Small        | Converges fast             |
| Go              | Massive      | Needs 100k+ simulations    |
| Grid navigation | Continuous   | Requires state abstraction |

#### Complexity

| Measure | Complexity                           |
| ------- | ------------------------------------ |
| Time    | $O(N \times D)$ (iterations × depth) |
| Space   | $O(N)$ (tree nodes)                  |

Each iteration grows the tree gradually.
Performance depends on how good the simulation policy is.

#### A Gentle Proof (Why It Works)

With enough iterations, MCTS converges to the optimal minimax value,
since all actions are explored infinitely often and their value estimates
approach expected returns by the law of large numbers.

The UCT rule ensures logarithmic balance between exploration and exploitation.

#### Summary Table

| Concept          | Description                                |
| ---------------- | ------------------------------------------ |
| Framework        | Monte Carlo sampling in tree search        |
| Core Idea        | Grow search tree by simulated play         |
| Exploration      | Upper Confidence Bound (UCT)               |
| Guarantee        | Converges to optimal policy asymptotically |
| Time Complexity  | $O(ND)$                                    |
| Space Complexity | $O(N)$                                     |
| Applications     | Games, planning, RL hybrids                |

MCTS learns not by seeing every path, but by *sampling wisely*.
Each simulation is a glimpse of the future, and together, they guide the search toward mastery.

### 997. Minimax Algorithm

The Minimax algorithm is the foundation of game-playing AI.
It models decision-making between two opponents, one trying to maximize the score, the other trying to minimize it.
Minimax explores the game tree and assumes both players play optimally.

It's used in Chess, Tic-Tac-Toe, Checkers, and any deterministic, turn-based, zero-sum game.

#### What Problem Are We Solving?

We want to choose the best possible move in a two-player game where each move alternates between maximizing and minimizing outcomes.

For a game tree with alternating turns:

- MAX tries to maximize the evaluation function.
- MIN tries to minimize it.

The final outcome is determined by recursively applying these opposing objectives down the tree.

#### The Core Idea

At each node (game state):

- If it's MAX's turn, choose the move with maximum value among children.
- If it's MIN's turn, choose the move with minimum value among children.

The value of a node is computed recursively:

$$
V(s) =
\begin{cases}
\text{Utility}(s), & \text{if } s \text{ is terminal},\\
\max_{a \in \text{Actions}(s)} V(\text{Result}(s,a)), & \text{if } s \text{ is MAX turn},\\
\min_{a \in \text{Actions}(s)} V(\text{Result}(s,a)), & \text{if } s \text{ is MIN turn.}
\end{cases}
$$


This yields an optimal strategy under perfect play.

#### Algorithm

| Step | Description                                                                |
| ---- | -------------------------------------------------------------------------- |
| 1    | Build or simulate the game tree to a fixed depth or until terminal states. |
| 2    | Assign evaluation scores to terminal nodes (win/loss/draw or heuristic).   |
| 3    | Propagate scores upward: alternate max/min at each level.                  |
| 4    | The root's best move is the child with the highest propagated score.       |

#### Tiny Code (Python Example)

```python
def minimax(state, depth, maximizing_player, evaluate, get_moves, make_move):
    if depth == 0 or not get_moves(state):
        return evaluate(state)

    if maximizing_player:
        max_eval = float('-inf')
        for move in get_moves(state):
            new_state = make_move(state, move)
            eval = minimax(new_state, depth - 1, False, evaluate, get_moves, make_move)
            max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = float('inf')
        for move in get_moves(state):
            new_state = make_move(state, move)
            eval = minimax(new_state, depth - 1, True, evaluate, get_moves, make_move)
            min_eval = min(min_eval, eval)
        return min_eval
```

Example (Tic-Tac-Toe heuristic):

```python
def evaluate(state):
    if state == 'win': return 1
    if state == 'lose': return -1
    return 0
```

#### Why It Matters

- Captures perfect rational play in deterministic games.
- Provides the foundation for Alpha–Beta pruning and MCTS.
- Shows the connection between search and strategic reasoning.

Used in:

- Turn-based games (Chess, Checkers, Connect Four)
- Simple planning with adversarial dynamics

#### Try It Yourself

1. Implement minimax for Tic-Tac-Toe.
2. Add a heuristic evaluator for non-terminal positions.
3. Compare performance with and without pruning.
4. Visualize the game tree expansion.

#### Test Cases

| Game             | Depth | Behavior       | Outcome              |
| ---------------- | ----- | -------------- | -------------------- |
| Tic-Tac-Toe      | full  | Optimal play   | Draw                 |
| Connect Four     | 4     | Approx optimal | Competitive play     |
| Simplified Chess | 2     | Greedy play    | Suboptimal but valid |

#### Complexity

| Measure | Complexity |
| ------- | ---------- |
| Time    | $O(b^d)$   |
| Space   | $O(bd)$    |

where $b$ is branching factor and $d$ is search depth.

#### A Gentle Proof (Why It Works)

Because both players are assumed optimal, each node's value represents the best achievable outcome for the current player, assuming the opponent also plays optimally.

By backward induction, this yields the Nash equilibrium of the two-player deterministic game.

#### Summary Table

| Concept          | Description                     |
| ---------------- | ------------------------------- |
| Framework        | Adversarial search              |
| Core Idea        | Alternate max and min decisions |
| Guarantee        | Optimal for perfect play        |
| Data Structure   | Game tree                       |
| Time Complexity  | $O(b^d)$                        |
| Space Complexity | $O(bd)$                         |
| Applications     | Board games, strategic planning |

Minimax plays the role of a perfect strategist, it doesn't react emotionally, doesn't gamble, just follows logic to its inevitable conclusion:
play perfectly, or be outplayed.

### 998. Alpha–Beta Pruning

Alpha–Beta Pruning is an optimization of the Minimax algorithm.
It prunes away branches that cannot affect the final decision, allowing us to search deeper without changing the result.

In essence, it tells the computer: "Don't explore that move, it can't possibly change the outcome."

#### What Problem Are We Solving?

Minimax explores every node in the game tree, even when some moves are already provably worse.
Alpha–Beta pruning avoids this by keeping track of two bounds:

- α (alpha), the best value the maximizer can guarantee so far.
- β (beta), the best value the minimizer can guarantee so far.

If at any point α ≥ β, we stop exploring that branch, it can't influence the decision.

#### The Core Idea

While traversing the tree:

- MAX nodes update α = max(α, value).
- MIN nodes update β = min(β, value).
- If α ≥ β, prune the rest of that branch (cutoff).

This reduces the number of evaluated nodes while preserving correctness.

#### Algorithm

| Step | Description                                       |
| ---- | ------------------------------------------------- |
| 1    | Initialize α = −∞, β = +∞.                        |
| 2    | Recursively evaluate game tree as in Minimax.     |
| 3    | For MAX nodes, update α; for MIN nodes, update β. |
| 4    | If α ≥ β, prune remaining children.               |
| 5    | Return α or β depending on player type.           |

#### Tiny Code (Python Example)

```python
def alphabeta(state, depth, alpha, beta, maximizing, evaluate, get_moves, make_move):
    if depth == 0 or not get_moves(state):
        return evaluate(state)

    if maximizing:
        value = float('-inf')
        for move in get_moves(state):
            new_state = make_move(state, move)
            value = max(value, alphabeta(new_state, depth-1, alpha, beta, False, evaluate, get_moves, make_move))
            alpha = max(alpha, value)
            if alpha >= beta:  # Beta cutoff
                break
        return value
    else:
        value = float('inf')
        for move in get_moves(state):
            new_state = make_move(state, move)
            value = min(value, alphabeta(new_state, depth-1, alpha, beta, True, evaluate, get_moves, make_move))
            beta = min(beta, value)
            if beta <= alpha:  # Alpha cutoff
                break
        return value
```

Example:

```python
score = alphabeta(start_state, 4, float('-inf'), float('inf'), True, evaluate, get_moves, make_move)
```

#### Why It Matters

- Same optimal result as Minimax, but with far fewer node evaluations.
- Enables deeper searches in complex games.
- Forms the basis for Chess, Checkers, and Go engines.

It's the standard search core of classic game AI.

#### Try It Yourself

1. Compare the number of nodes expanded by Minimax vs Alpha–Beta.
2. Implement move ordering, see how pruning improves dramatically.
3. Apply it to Tic-Tac-Toe, Connect Four, or small Chess subsets.
4. Add a transposition table to skip repeated states.

#### Test Cases

| Game         | Depth | Algorithm  | Nodes Evaluated  |
| ------------ | ----- | ---------- | ---------------- |
| Tic-Tac-Toe  | 6     | Minimax    | ~550             |
| Tic-Tac-Toe  | 6     | Alpha–Beta | ~80              |
| Connect Four | 6     | Alpha–Beta | ~40% fewer nodes |

#### Complexity

| Measure | Complexity                                 |
| ------- | ------------------------------------------ |
| Time    | $O(b^{m})$ best-case $O(b^{d})$ worst-case |
| Space   | $O(bd)$                                    |

where:

- $b$ = branching factor
- $d$ = search depth
- $m$ = effective depth after pruning (depends on move order)

#### A Gentle Proof (Why It Works)

If α ≥ β at a node, it means the current player has already found a better move elsewhere,
and this branch cannot improve the outcome.
Thus pruning is safe, the optimal value remains unchanged.

Formally, Alpha–Beta preserves the backward induction property of Minimax.

#### Summary Table

| Concept          | Description                              |
| ---------------- | ---------------------------------------- |
| Framework        | Optimized adversarial search             |
| Core Idea        | Prune branches that can't affect outcome |
| Guarantee        | Identical result to Minimax              |
| Key Variables    | α (best for MAX), β (best for MIN)       |
| Time Complexity  | $O(b^m)$ (with pruning)                  |
| Space Complexity | $O(bd)$                                  |
| Applications     | Game engines, decision-making AI         |

Alpha–Beta pruning is Minimax made efficient, 
a master strategist that knows when not to waste time thinking about moves that don't matter.

### 999. STRIPS Planning

STRIPS (Stanford Research Institute Problem Solver) is a foundational algorithm in AI planning, designed to automatically generate sequences of actions that lead from an initial state to a desired goal.

It was developed in the early 1970s and remains the conceptual backbone of modern planners and symbolic reasoning systems.

#### What Problem Are We Solving?

We want an algorithm that plans actions to achieve a goal, not just reactively act.

Given:

- An initial state (facts about the world)
- A goal (conditions to be made true)
- A set of actions, each with:

  * Preconditions (what must be true to apply the action)
  * Add effects (facts made true after action)
  * Delete effects (facts made false after action)

STRIPS systematically searches for a sequence of actions that transform the initial state into one satisfying the goal.

#### The Core Idea

STRIPS works through symbolic state-space search, applying operators that change world facts.

Each action is defined as:

$$
\text{Action}(A) = \langle \text{Pre}(A), \text{Add}(A), \text{Del}(A) \rangle
$$

When applied to a state $S$:

$$
\text{Result}(S, A) = (S - \text{Del}(A)) \cup \text{Add}(A)
$$

The planner uses forward search (from start to goal) or backward search (from goal to start)
to find a valid plan, a sequence of actions $\langle A_1, A_2, ..., A_n \rangle$
that leads to the goal.

#### Example

Let's say we want to move a robot from Room1 to Room2:

- Initial state: In(Room1)
- Goal: In(Room2)
- Action: Move(Room1, Room2)

| Action     | Preconditions | Add Effects | Delete Effects |
| ---------- | ------------- | ----------- | -------------- |
| Move(x, y) | In(x)         | In(y)       | In(x)          |

Forward search applies this operator:

$$
S_0 = { In(Room1) }
$$
$$
Move(Room1, Room2):\quad S_1 = (S_0 - { In(Room1) }) \cup { In(Room2) } = { In(Room2) }
$$

Goal achieved.

#### Algorithm Outline

| Step | Description                                                  |
| ---- | ------------------------------------------------------------ |
| 1    | Represent initial state, goal, and actions using predicates. |
| 2    | Start from the initial state.                                |
| 3    | Choose applicable actions (whose preconditions are true).    |
| 4    | Apply action to generate successor states.                   |
| 5    | Continue until the goal condition is satisfied.              |

Can use BFS, DFS, or heuristic-based search (like A*).

#### Tiny Code (Python Example)

```python
from collections import deque

def strips_plan(initial, goal, actions):
    queue = deque([(initial, [])])
    while queue:
        state, plan = queue.popleft()
        if goal <= state:  # all goal facts satisfied
            return plan
        for name, pre, add, delete in actions:
            if pre <= state:
                new_state = (state - delete) | add
                queue.append((new_state, plan + [name]))
    return None

# Example usage
actions = [
    ("Move(Room1, Room2)", {"In(Room1)"}, {"In(Room2)"}, {"In(Room1)"}),
$$

plan = strips_plan({"In(Room1)"}, {"In(Room2)"}, actions)
print(plan)  # ['Move(Room1, Room2)']
```

#### Why It Matters

- Introduced formal symbolic planning, the ability for machines to reason about *what must be done*, not just *what to do now*.
- Basis for PDDL (Planning Domain Definition Language), automated theorem proving, and robot motion planning.
- Still used conceptually in modern AI planners and reinforcement learning with symbolic models.

#### Try It Yourself

1. Model a block-stacking problem (Block A on B, B on C, etc.).
2. Implement backward search (regression) planning.
3. Add multiple actions and test branching.
4. Compare STRIPS planning with A* search on same domain.

#### Test Cases

| Domain    | Initial                              | Goal               | Plan                   |
| --------- | ------------------------------------ | ------------------ | ---------------------- |
| Move      | In(Room1)                            | In(Room2)          | [Move(Room1, Room2)]   |
| Blocks    | On(A, Table), Clear(B)               | On(A, B)           | [Pick(A), Stack(A, B)] |
| Logistics | In(Truck, Depot), At(Package, Depot) | Delivered(Package) | [Load, Drive, Unload]  |

#### Complexity

| Measure | Complexity                               |
| ------- | ---------------------------------------- |
| Time    | Exponential in number of actions         |
| Space   | Exponential in state representation size |

#### A Gentle Proof (Why It Works)

STRIPS is sound and complete for domains expressible in propositional logic:
If a plan exists that satisfies the goal, STRIPS will eventually find it,
because all applicable actions are systematically explored.

It encodes the world as logical transitions, ensuring each step preserves consistency.

#### Summary Table

| Concept          | Description                             |
| ---------------- | --------------------------------------- |
| Framework        | Symbolic AI planning                    |
| Core Idea        | Search using logical action operators   |
| Guarantee        | Sound and complete (finite domains)     |
| Data Structure   | States as sets of predicates            |
| Time Complexity  | Exponential                             |
| Space Complexity | Exponential                             |
| Applications     | Robotics, planning, automated reasoning |

STRIPS marked the dawn of AI as reasoning, 
machines thinking in terms of *what must be true* and *what must change*
to make the world match a goal.

### 1000. Hierarchical Task Network (HTN) Planning

Hierarchical Task Network (HTN) Planning extends classical STRIPS planning by adding *structure* to how plans are formed.
Instead of searching through flat, atomic actions, HTN organizes them into tasks, some abstract (high-level) and others primitive (executable).
It mirrors how humans think: *"To travel, first pack, then go to the airport, then fly."*

#### What Problem Are We Solving?

Classical STRIPS is good at exploring all possible actions, but it doesn't scale well or encode knowledge about *how tasks are usually done*.
HTN Planning introduces hierarchical decomposition, letting planners break complex goals into manageable subtasks guided by domain knowledge.

Given:

- An initial state
- A set of tasks, where

  * Primitive tasks are executable actions
  * Compound tasks are decomposed into subtasks using methods

The planner recursively refines the main goal task into smaller actions until only primitive steps remain.

#### The Core Idea

Each method tells the planner *how* to achieve a compound task, depending on the current context.

Let:

- $T$ = set of tasks
- $M$ = set of methods
- $A$ = set of primitive actions

A method $m$ is defined as:

$$
m = \langle \text{name}(m), \text{task}(m), \text{subtasks}(m), \text{preconditions}(m) \rangle
$$

The planner starts from a top-level task $t_0$, recursively decomposes it using applicable methods,
and continues until the plan consists entirely of executable actions.

#### Example

Let's plan Deliver(Package, Destination):

Initial state:
`At(Truck, Depot), At(Package, Depot)`

Goal task:
`Deliver(Package, Destination)`

Methods:

| Method           | Preconditions                | Subtasks                                                                     |
| ---------------- | ---------------------------- | ---------------------------------------------------------------------------- |
| Deliver-by-truck | At(Truck, x), At(Package, x) | [Load(Package, Truck), Drive(Truck, x, Destination), Unload(Package, Truck)] |

Primitive actions (STRIPS-style):

- `Load(p, t)`: Pre {At(p, x), At(t, x)}, Add {In(p, t)}, Del {At(p, x)}
- `Drive(t, x, y)`: Pre {At(t, x)}, Add {At(t, y)}, Del {At(t, x)}
- `Unload(p, t)`: Pre {In(p, t), At(t, y)}, Add {At(p, y)}, Del {In(p, t)}

The decomposition leads to the final executable plan:

1. Load(Package, Truck)
2. Drive(Truck, Depot, Destination)
3. Unload(Package, Truck)

#### Algorithm Outline

| Step | Description                                                                                         |
| ---- | --------------------------------------------------------------------------------------------------- |
| 1    | Start with a task network (list of tasks to achieve).                                               |
| 2    | Pick a task: if primitive, check preconditions and execute; if compound, find an applicable method. |
| 3    | Replace the compound task with its subtasks.                                                        |
| 4    | Repeat until only primitive actions remain.                                                         |
| 5    | Execute primitive actions to reach the goal.                                                        |

#### Tiny Code (Python Example)

```python
def htn_plan(tasks, methods, actions, state):
    if not tasks:
        return []
    task = tasks[0]
    rest = tasks[1:]
    # Primitive task
    if task in actions:
        pre, add, delete = actions[task]
        if pre <= state:
            new_state = (state - delete) | add
            return [task] + htn_plan(rest, methods, actions, new_state)
        else:
            return None
    # Compound task
    for name, goal, subtasks, pre in methods:
        if goal == task and pre <= state:
            new_tasks = subtasks + rest
            plan = htn_plan(new_tasks, methods, actions, state)
            if plan:
                return plan
    return None
```

Example domain setup:

```python
actions = {
    "Load": ({"At(Package, Depot)", "At(Truck, Depot)"}, {"In(Package, Truck)"}, {"At(Package, Depot)"}),
    "Drive": ({"At(Truck, Depot)"}, {"At(Truck, Destination)"}, {"At(Truck, Depot)"}),
    "Unload": ({"In(Package, Truck)", "At(Truck, Destination)"}, {"At(Package, Destination)"}, {"In(Package, Truck)"})
}

methods = [
    ("Deliver-by-truck", "Deliver", ["Load", "Drive", "Unload"], {"At(Truck, Depot)", "At(Package, Depot)"})
$$

plan = htn_plan(["Deliver"], methods, actions, {"At(Truck, Depot)", "At(Package, Depot)"})
print(plan)
```

Output:

```
$$'Load', 'Drive', 'Unload']
```

#### Why It Matters

- Encodes domain knowledge explicitly, how tasks are normally decomposed.
- Enables scalable planning for complex systems (robotics, logistics, simulation).
- Bridges the gap between symbolic planning and real-world procedural knowledge.

Used in:

- Robotics (multi-step manipulation tasks)
- Game AI (unit control, story planning)
- Simulation engines (multi-agent coordination)

#### Try It Yourself

1. Build an HTN planner for a household robot.
2. Add methods for *CookMeal*, *CleanRoom*, *MakeTea*.
3. Visualize decomposition trees.
4. Compare plan lengths with and without hierarchical guidance.

#### Test Cases

| Domain    | Task             | Plan                      |
| --------- | ---------------- | ------------------------- |
| Logistics | Deliver(Package) | [Load, Drive, Unload]     |
| Household | MakeTea          | [BoilWater, AddTea, Pour] |
| Game AI   | AttackEnemy      | [MoveToTarget, Aim, Fire] |

#### Complexity

| Measure | Complexity                                    |
| ------- | --------------------------------------------- |
| Time    | Exponential (depends on branching of methods) |
| Space   | Linear in decomposition depth                 |

#### A Gentle Proof (Why It Works)

HTN preserves soundness:
if every method is valid (its subtasks achieve its parent's goal), then the entire plan is valid.

It's not *complete* (it may miss valid plans if no applicable methods exist),
but it's far more *practical*, guiding search with human-like structure.

#### Summary Table

| Concept          | Description                                        |
| ---------------- | -------------------------------------------------- |
| Framework        | Hierarchical symbolic planning                     |
| Core Idea        | Decompose complex tasks into simpler subtasks      |
| Guarantee        | Sound but not complete                             |
| Data Structure   | Task network                                       |
| Time Complexity  | Exponential                                        |
| Space Complexity | Linear                                             |
| Applications     | Robotics, simulation, game AI, workflow automation |

HTN planning closes our 1000-algorithm journey, 
from sorting numbers to organizing intelligent behavior.
It shows how structure, knowledge, and recursion combine to turn reasoning into action, 
and action into purpose.

