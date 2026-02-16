# Hierarchical Clustering from Scratch - Step by Step Guide

## Overview

This guide walks you through implementing hierarchical clustering from scratch, covering all mathematical foundations and implementation details.

---

## Table of Contents

1. [Mathematical Foundation](#1-mathematical-foundation)
2. [Step 1: Distance Calculations](#step-1-distance-calculations)
3. [Step 2: Linkage Criteria](#step-2-linkage-criteria)
4. [Step 3: Agglomerative Algorithm](#step-3-agglomerative-algorithm)
5. [Step 4: Dendrogram Construction](#step-4-dendrogram-construction)
6. [Step 5: Cluster Extraction](#step-5-cluster-extraction)
7. [Complexity Analysis](#complexity-analysis)
8. [Usage Examples](#usage-examples)

---

## 1. Mathematical Foundation

### What is Hierarchical Clustering?

Hierarchical clustering is an unsupervised learning algorithm that builds a hierarchy of clusters. It creates a tree-like structure (dendrogram) showing how data points group together at different levels of similarity.

### Two Approaches:

1. **Agglomerative (Bottom-Up)** ← We'll implement this
   - Start: Each point is its own cluster
   - Repeat: Merge the two closest clusters
   - End: All points in one cluster

2. **Divisive (Top-Down)**
   - Start: All points in one cluster
   - Repeat: Split the most diverse cluster
   - End: Each point is its own cluster

### Key Advantages:

- ✓ No need to specify number of clusters beforehand
- ✓ Produces a dendrogram showing cluster hierarchy
- ✓ Works with non-spherical clusters
- ✓ Deterministic results (no random initialization)

### Key Limitations:

- ✗ High computational complexity: O(N³)
- ✗ Sensitive to noise and outliers
- ✗ Once merged, clusters cannot be separated
- ✗ Curse of dimensionality in high dimensions

---

## Step 1: Distance Calculations

### Euclidean Distance

The foundation of hierarchical clustering is measuring similarity between points.

**Formula:**
```
d(p, q) = √(Σ(p_i - q_i)²)
```

**Implementation:**
```python
def euclidean_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt(np.sum((point1 - point2) ** 2))
```

### Distance Matrix

Create an N × N matrix storing all pairwise distances.

**Properties:**
- Symmetric: d(i,j) = d(j,i)
- Diagonal zeros: d(i,i) = 0
- Non-negative: d(i,j) ≥ 0

**Implementation:**
```python
def compute_distance_matrix(X):
    """Compute pairwise distance matrix for all points."""
    n_samples = X.shape[0]
    distance_matrix = np.zeros((n_samples, n_samples))
    
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            dist = euclidean_distance(X[i], X[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist  # Exploit symmetry
    
    return distance_matrix
```

**Complexity:** O(N² × d) where d is dimensionality

---

## Step 2: Linkage Criteria

Linkage defines how we measure distance between **clusters** (not just points).

### 1. Single Linkage (MIN / Nearest Neighbor)

**Definition:** Minimum distance between any two points from different clusters

**Formula:**
```
d(C₁, C₂) = min{d(p, q) : p ∈ C₁, q ∈ C₂}
```

**Characteristics:**
- Creates elongated, "chaining" clusters
- Sensitive to noise
- Good for non-spherical clusters

**Implementation:**
```python
def single_linkage(cluster1, cluster2, distance_matrix):
    """Minimum distance between any two points."""
    min_dist = np.inf
    for i in cluster1:
        for j in cluster2:
            if distance_matrix[i, j] < min_dist:
                min_dist = distance_matrix[i, j]
    return min_dist
```

### 2. Complete Linkage (MAX / Furthest Neighbor)

**Definition:** Maximum distance between any two points from different clusters

**Formula:**
```
d(C₁, C₂) = max{d(p, q) : p ∈ C₁, q ∈ C₂}
```

**Characteristics:**
- Creates compact, spherical clusters
- Less sensitive to outliers
- Prefers similar-sized clusters

**Implementation:**
```python
def complete_linkage(cluster1, cluster2, distance_matrix):
    """Maximum distance between any two points."""
    max_dist = 0
    for i in cluster1:
        for j in cluster2:
            if distance_matrix[i, j] > max_dist:
                max_dist = distance_matrix[i, j]
    return max_dist
```

### 3. Average Linkage (UPGMA)

**Definition:** Average distance between all pairs of points from different clusters

**Formula:**
```
d(C₁, C₂) = (1 / |C₁| × |C₂|) × Σ d(p, q)
```

**Characteristics:**
- Balanced approach
- Moderately robust
- Good general-purpose choice

**Implementation:**
```python
def average_linkage(cluster1, cluster2, distance_matrix):
    """Average distance between all pairs of points."""
    total_dist = 0
    count = 0
    for i in cluster1:
        for j in cluster2:
            total_dist += distance_matrix[i, j]
            count += 1
    return total_dist / count if count > 0 else 0
```

### Comparison Table

| Linkage | Shape Preference | Outlier Sensitivity | Complexity | Use Case |
|---------|-----------------|---------------------|------------|----------|
| Single | Elongated | High | Can optimize to O(N²) | Non-spherical clusters |
| Complete | Spherical | Low | O(N³) | Compact, similar-sized clusters |
| Average | Balanced | Medium | O(N³) | General purpose |

---

## Step 3: Agglomerative Algorithm

The core algorithm implements bottom-up merging.

### Pseudocode:

```
1. Initialize: Each point is its own cluster (N clusters)
2. Compute: Distance matrix for all points
3. Repeat N-1 times:
   a. Find two closest clusters using linkage criterion
   b. Merge them into a new cluster
   c. Record merge (cluster IDs, distance, size)
   d. Update cluster list
4. Return: Linkage matrix (merge history)
```

### Detailed Implementation:

```python
class HierarchicalClustering:
    def __init__(self, linkage='single'):
        self.linkage = linkage
        self.linkage_func = {
            'single': single_linkage,
            'complete': complete_linkage,
            'average': average_linkage
        }[linkage]
        
    def fit(self, X):
        n_samples = X.shape[0]
        
        # Step 1: Compute distance matrix
        distance_matrix = compute_distance_matrix(X)
        
        # Step 2: Initialize clusters
        clusters = {i: [i] for i in range(n_samples)}
        
        # Step 3: Track merge history
        linkage_matrix = []
        next_cluster_id = n_samples
        
        # Step 4: Iteratively merge
        while len(clusters) > 1:
            # Find two closest clusters
            min_dist = np.inf
            merge_pair = None
            
            cluster_ids = list(clusters.keys())
            for i, id1 in enumerate(cluster_ids):
                for id2 in cluster_ids[i+1:]:
                    dist = self.linkage_func(
                        clusters[id1], 
                        clusters[id2], 
                        distance_matrix
                    )
                    if dist < min_dist:
                        min_dist = dist
                        merge_pair = (id1, id2)
            
            # Merge
            id1, id2 = merge_pair
            new_cluster = clusters[id1] + clusters[id2]
            
            # Record
            linkage_matrix.append([
                id1, id2, min_dist, len(new_cluster)
            ])
            
            # Update
            del clusters[id1]
            del clusters[id2]
            clusters[next_cluster_id] = new_cluster
            next_cluster_id += 1
        
        self.linkage_matrix_ = np.array(linkage_matrix)
        return self
```

---

## Step 4: Dendrogram Construction

The linkage matrix encodes the dendrogram structure.

### Linkage Matrix Format:

Each row represents one merge:
```
[cluster1_id, cluster2_id, distance, new_cluster_size]
```

**Example:**
```
Step 1: [0, 5, 0.234, 2]  → Merge points 0 and 5, distance=0.234
Step 2: [1, 3, 0.456, 2]  → Merge points 1 and 3, distance=0.456
Step 3: [50, 51, 0.789, 4] → Merge clusters 50 and 51, distance=0.789
```

### Visualization with scipy:

```python
from scipy.cluster.hierarchy import dendrogram

dendrogram(linkage_matrix)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Data Point Index')
plt.ylabel('Distance')
plt.show()
```

**Reading a Dendrogram:**
- X-axis: Data point indices
- Y-axis: Distance at which clusters merge
- Horizontal line: Represents a merge
- Height of line: Distance between merged clusters
- Cutting horizontally: Determines number of clusters

---

## Step 5: Cluster Extraction

Extract flat clusters by "cutting" the dendrogram.

### Method: Cut at specific number of clusters

```python
def get_clusters(self, n_clusters):
    """Extract n_clusters from the hierarchy."""
    # Access history at specific step
    cluster_state = self.clusters_history_[-n_clusters]
    
    # Assign labels
    n_samples = len(self.clusters_history_[0])
    labels = np.zeros(n_samples, dtype=int)
    
    for cluster_label, (cluster_id, members) in enumerate(cluster_state.items()):
        for point_idx in members:
            labels[point_idx] = cluster_label
    
    return labels
```

### Example:

For 50 points wanting 3 clusters:
- Access: `clusters_history_[-3]` (3 clusters from end)
- This gives us the state when exactly 3 clusters existed
- Map each point to its cluster label

---

## Complexity Analysis

### Time Complexity

**Naive Implementation:** O(N³)
- Computing distance matrix: O(N²)
- N-1 iterations
- Each iteration:
  - Finding minimum: O(N²) comparisons
  - Total: O(N³)

**Optimized Implementations:**

| Algorithm | Linkage | Complexity | Method |
|-----------|---------|------------|--------|
| SLINK | Single | O(N²) | Pointer jumping |
| CLINK | Complete | O(N²) | Similar to SLINK |
| Priority Queue | Any | O(N² log N) | Heap for distances |

### Space Complexity

- Distance matrix: O(N²)
- Cluster storage: O(N)
- Linkage matrix: O(N)
- **Total: O(N²)**

### Practical Considerations

**Dataset Size Guidelines:**
- N < 1,000: Fast (< 1 second)
- 1,000 < N < 5,000: Moderate (seconds to minutes)
- N > 10,000: Slow (consider alternatives)

**High Dimensionality:**
- Curse of dimensionality affects distance metrics
- Consider dimensionality reduction (PCA, t-SNE)
- Or use density-based methods (DBSCAN, HDBSCAN)

---

## Usage Examples

### Basic Usage

```python
from hierarchical_clustering_scratch import HierarchicalClustering
from sklearn.datasets import make_blobs

# Generate data
X, y = make_blobs(n_samples=100, n_features=2, centers=3)

# Fit clustering
hc = HierarchicalClustering(linkage='average')
hc.fit(X)

# Get 3 clusters
labels = hc.get_clusters(n_clusters=3)

# Visualize
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.show()
```

### Compare Linkage Methods

```python
for linkage in ['single', 'complete', 'average']:
    hc = HierarchicalClustering(linkage=linkage)
    hc.fit(X)
    labels = hc.get_clusters(3)
    
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.title(f'{linkage.capitalize()} Linkage')
    plt.show()
```

### Dendrogram Analysis

```python
from scipy.cluster.hierarchy import dendrogram

hc = HierarchicalClustering(linkage='average')
hc.fit(X)

plt.figure(figsize=(12, 6))
dendrogram(hc.get_linkage_matrix())
plt.title('Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()
```

### Validate Against sklearn

```python
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

# Our implementation
hc_custom = HierarchicalClustering(linkage='average')
hc_custom.fit(X)
labels_custom = hc_custom.get_clusters(3)

# sklearn
hc_sklearn = AgglomerativeClustering(n_clusters=3, linkage='average')
labels_sklearn = hc_sklearn.fit_predict(X)

# Compare
ari = adjusted_rand_score(labels_custom, labels_sklearn)
print(f"Agreement with sklearn: {ari:.4f}")
```

---

## Advanced Topics

### 1. Lance-Williams Update Formula

Instead of recomputing all distances, use recursive formula:

```
d(C_k, C_{i,j}) = α_i × d(C_k, C_i) + α_j × d(C_k, C_j) + β × d(C_i, C_j) + γ × |d(C_k, C_i) - d(C_k, C_j)|
```

Parameters vary by linkage:
- Single: α_i=α_j=0.5, β=0, γ=-0.5
- Complete: α_i=α_j=0.5, β=0, γ=0.5
- Average: α_i=n_i/(n_i+n_j), α_j=n_j/(n_i+n_j), β=γ=0

### 2. Handling Ties

When multiple cluster pairs have same minimum distance:
- Use consistent tie-breaking (e.g., lower indices first)
- Ensures deterministic results

### 3. Distance Metrics

Can use other metrics besides Euclidean:
- Manhattan: `Σ|p_i - q_i|`
- Cosine: `1 - (p·q)/(||p|| × ||q||)`
- Mahalanobis: Accounts for covariance

### 4. Dendrogram Cutting Strategies

**By distance threshold:**
```python
threshold = 5.0
labels = fcluster(linkage_matrix, threshold, criterion='distance')
```

**By inconsistency:**
```python
labels = fcluster(linkage_matrix, 2.0, criterion='inconsistent')
```

---

## Summary Checklist

✓ **Understand** distance metrics and linkage criteria  
✓ **Implement** pairwise distance computation  
✓ **Code** the agglomerative merging algorithm  
✓ **Track** merge history in linkage matrix  
✓ **Extract** flat clusters at desired granularity  
✓ **Visualize** using dendrograms  
✓ **Validate** against established implementations  
✓ **Analyze** computational complexity  

---

## Further Reading

1. **Papers:**
   - Sibson, R. (1973). "SLINK: An optimally efficient algorithm for the single-link cluster method"
   - Defays, D. (1977). "An efficient algorithm for complete linkage method"

2. **Books:**
   - "The Elements of Statistical Learning" - Hastie, Tibshirani, Friedman
   - "Introduction to Data Mining" - Tan, Steinbach, Kumar

3. **Online Resources:**
   - Scikit-learn documentation: Hierarchical clustering
   - SciPy hierarchy module documentation

---

## Files in This Implementation

1. `hierarchical_clustering_scratch.py` - Full implementation
2. `STEP_BY_STEP_GUIDE.md` - This guide
3. `lab.ipynb` - Interactive notebook (coming next)

---

**Ready to implement? Run:**

```bash
python hierarchical_clustering_scratch.py
```

This will execute a complete demonstration with visualizations!
