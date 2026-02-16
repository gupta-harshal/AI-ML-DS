"""
Hierarchical Clustering from Scratch
=====================================

A complete implementation of agglomerative hierarchical clustering algorithm
demonstrating the mathematical foundations and computational principles.

Author: Implementation for Educational Purposes
Date: February 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import time
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# PART 1: DISTANCE CALCULATIONS
# ============================================================================

def euclidean_distance(point1, point2):
    """
    Calculate Euclidean distance between two points.
    
    Formula: d(p, q) = sqrt(sum((p_i - q_i)^2))
    
    Parameters:
    -----------
    point1, point2 : array-like
        Two points in n-dimensional space
        
    Returns:
    --------
    float : Euclidean distance
    
    Time Complexity: O(d) where d is the number of dimensions
    """
    return np.sqrt(np.sum((point1 - point2) ** 2))


def compute_distance_matrix(X):
    """
    Compute pairwise distance matrix for all points.
    
    Creates a symmetric N x N matrix where entry (i, j) represents
    the distance between points i and j.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data
        
    Returns:
    --------
    distance_matrix : ndarray, shape (n_samples, n_samples)
        Symmetric distance matrix
        
    Time Complexity: O(N^2 * d) where N is number of samples, d is dimensions
    Space Complexity: O(N^2)
    """
    n_samples = X.shape[0]
    distance_matrix = np.zeros((n_samples, n_samples))
    
    # Compute upper triangle (matrix is symmetric)
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            dist = euclidean_distance(X[i], X[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist  # Exploit symmetry
    
    return distance_matrix


# ============================================================================
# PART 2: LINKAGE CRITERIA
# ============================================================================

def single_linkage(cluster1, cluster2, distance_matrix):
    """
    Single Linkage (Nearest Neighbor).
    
    Distance between clusters is defined as the minimum distance
    between any two points from the two clusters.
    
    Formula: d(C1, C2) = min{d(p, q) : p ∈ C1, q ∈ C2}
    
    Properties:
    - Tends to create elongated, "chaining" clusters
    - Sensitive to noise and outliers
    - Can handle non-spherical clusters
    
    Parameters:
    -----------
    cluster1, cluster2 : list
        Lists of point indices belonging to each cluster
    distance_matrix : ndarray
        Pairwise distance matrix
        
    Returns:
    --------
    float : Minimum distance between the clusters
    """
    min_dist = np.inf
    for i in cluster1:
        for j in cluster2:
            dist = distance_matrix[i, j]
            if dist < min_dist:
                min_dist = dist
    return min_dist


def complete_linkage(cluster1, cluster2, distance_matrix):
    """
    Complete Linkage (Furthest Neighbor).
    
    Distance between clusters is defined as the maximum distance
    between any two points from the two clusters.
    
    Formula: d(C1, C2) = max{d(p, q) : p ∈ C1, q ∈ C2}
    
    Properties:
    - Tends to create compact, spherical clusters
    - Less sensitive to outliers than single linkage
    - Prefers clusters of similar diameter
    
    Parameters:
    -----------
    cluster1, cluster2 : list
        Lists of point indices belonging to each cluster
    distance_matrix : ndarray
        Pairwise distance matrix
        
    Returns:
    --------
    float : Maximum distance between the clusters
    """
    max_dist = 0
    for i in cluster1:
        for j in cluster2:
            dist = distance_matrix[i, j]
            if dist > max_dist:
                max_dist = dist
    return max_dist


def average_linkage(cluster1, cluster2, distance_matrix):
    """
    Average Linkage (UPGMA - Unweighted Pair Group Method with Arithmetic Mean).
    
    Distance between clusters is defined as the average distance
    between all pairs of points from the two clusters.
    
    Formula: d(C1, C2) = (1 / |C1| * |C2|) * sum{d(p, q) : p ∈ C1, q ∈ C2}
    
    Properties:
    - Balanced approach between single and complete linkage
    - More robust to outliers than single linkage
    - Creates moderately compact clusters
    
    Parameters:
    -----------
    cluster1, cluster2 : list
        Lists of point indices belonging to each cluster
    distance_matrix : ndarray
        Pairwise distance matrix
        
    Returns:
    --------
    float : Average distance between the clusters
    """
    total_dist = 0
    count = 0
    for i in cluster1:
        for j in cluster2:
            total_dist += distance_matrix[i, j]
            count += 1
    return total_dist / count if count > 0 else 0


# ============================================================================
# PART 3: HIERARCHICAL CLUSTERING ALGORITHM
# ============================================================================

class HierarchicalClustering:
    """
    Agglomerative Hierarchical Clustering from Scratch.
    
    Implements bottom-up clustering where each observation starts in its own
    cluster and pairs of clusters are merged as one moves up the hierarchy.
    
    Algorithm:
    ----------
    1. Start with N clusters (one per data point)
    2. Compute distance matrix
    3. Repeat N-1 times:
        a. Find two closest clusters
        b. Merge them into a new cluster
        c. Update distance matrix
        d. Record merge in linkage matrix
    
    Parameters:
    -----------
    linkage : str, default='single'
        Linkage criterion: 'single', 'complete', or 'average'
        
    Attributes:
    -----------
    labels_ : array, shape (n_samples,)
        Cluster labels for each point
    linkage_matrix_ : array, shape (n_samples-1, 4)
        Records the merges in format [cluster1, cluster2, distance, size]
    clusters_history_ : list
        History of cluster states at each merge step
        
    Complexity:
    -----------
    Time: O(N^3) for naive implementation
          - N-1 iterations
          - Each iteration: O(N^2) to find minimum distance
    Space: O(N^2) for distance matrix
    
    Optimizations (not implemented here):
    ------------------------------------
    - Priority queue: Reduces time to O(N^2 log N)
    - SLINK algorithm: O(N^2) for single linkage
    - CLINK algorithm: O(N^2) for complete linkage
    """
    
    def __init__(self, linkage='single'):
        self.linkage = linkage
        self.linkage_functions = {
            'single': single_linkage,
            'complete': complete_linkage,
            'average': average_linkage
        }
        
        if linkage not in self.linkage_functions:
            raise ValueError(f"Linkage must be one of {list(self.linkage_functions.keys())}")
        
        self.linkage_func = self.linkage_functions[linkage]
        self.labels_ = None
        self.linkage_matrix_ = None
        self.clusters_history_ = []
    
    def fit(self, X):
        """
        Perform hierarchical clustering on dataset X.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
            
        Returns:
        --------
        self : object
            Returns the instance itself
        """
        n_samples = X.shape[0]
        
        print(f"Starting Hierarchical Clustering with {self.linkage} linkage...")
        print(f"Data shape: {X.shape}")
        
        # Step 1: Compute distance matrix
        print("Step 1: Computing distance matrix...")
        distance_matrix = compute_distance_matrix(X)
        print(f"  Distance matrix: {distance_matrix.shape}")
        
        # Step 2: Initialize clusters - each point is its own cluster
        print("Step 2: Initializing clusters...")
        clusters = {i: [i] for i in range(n_samples)}
        self.clusters_history_.append(clusters.copy())
        print(f"  Initial clusters: {n_samples}")
        
        # Step 3: Initialize linkage matrix for dendrogram
        # Format: [cluster1_id, cluster2_id, distance, size]
        self.linkage_matrix_ = []
        
        # Next cluster ID (starts after original data points)
        next_cluster_id = n_samples
        
        # Step 4: Iteratively merge clusters
        print("Step 3: Merging clusters...")
        iteration = 0
        
        while len(clusters) > 1:
            iteration += 1
            
            # Find the pair of clusters with minimum distance
            min_dist = np.inf
            merge_pair = None
            
            cluster_ids = list(clusters.keys())
            for i, id1 in enumerate(cluster_ids):
                for id2 in cluster_ids[i + 1:]:
                    dist = self.linkage_func(clusters[id1], clusters[id2], distance_matrix)
                    if dist < min_dist:
                        min_dist = dist
                        merge_pair = (id1, id2)
            
            # Merge the closest pair
            id1, id2 = merge_pair
            new_cluster = clusters[id1] + clusters[id2]
            
            if iteration <= 5 or iteration % 10 == 0:
                print(f"  Iteration {iteration}: Merging clusters {id1} and {id2} "
                      f"(distance={min_dist:.4f}, new size={len(new_cluster)})")
            
            # Record the merge in linkage matrix
            self.linkage_matrix_.append([
                id1, 
                id2, 
                min_dist, 
                len(new_cluster)
            ])
            
            # Remove old clusters and add new one
            del clusters[id1]
            del clusters[id2]
            clusters[next_cluster_id] = new_cluster
            next_cluster_id += 1
            
            # Save history
            self.clusters_history_.append(clusters.copy())
        
        self.linkage_matrix_ = np.array(self.linkage_matrix_)
        print(f"Clustering complete! Total merges: {len(self.linkage_matrix_)}")
        
        return self
    
    def get_clusters(self, n_clusters):
        """
        Cut the dendrogram to get specified number of clusters.
        
        Parameters:
        -----------
        n_clusters : int
            Number of desired clusters
            
        Returns:
        --------
        labels : array, shape (n_samples,)
            Cluster labels for each point
        """
        if n_clusters < 1:
            raise ValueError("Number of clusters must be at least 1")
        
        if n_clusters > len(self.clusters_history_[0]):
            raise ValueError(f"Number of clusters cannot exceed {len(self.clusters_history_[0])}")
        
        # Get the clustering state from history
        # Index: -n_clusters gives us n_clusters configuration
        cluster_state = self.clusters_history_[-n_clusters]
        
        # Assign labels
        n_samples = len(self.clusters_history_[0])
        labels = np.zeros(n_samples, dtype=int)
        
        for cluster_label, (cluster_id, members) in enumerate(cluster_state.items()):
            for point_idx in members:
                labels[point_idx] = cluster_label
        
        self.labels_ = labels
        return labels
    
    def get_linkage_matrix(self):
        """
        Return the linkage matrix for dendrogram visualization.
        
        Returns:
        --------
        linkage_matrix_ : array, shape (n_samples-1, 4)
            Linkage matrix in format [cluster1, cluster2, distance, size]
        """
        return self.linkage_matrix_


# ============================================================================
# PART 4: VISUALIZATION UTILITIES
# ============================================================================

def plot_clusters(X, labels, title="Hierarchical Clustering Results", ax=None):
    """
    Visualize clustering results in 2D.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, 2)
        2D data points
    labels : array-like, shape (n_samples,)
        Cluster labels
    title : str
        Plot title
    ax : matplotlib axis, optional
        Axis to plot on
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, s=100, 
                        cmap='viridis', edgecolors='black', alpha=0.7)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.grid(True, alpha=0.3)
    
    return scatter


def plot_dendrogram(linkage_matrix, title="Dendrogram", ax=None):
    """
    Plot dendrogram from linkage matrix.
    
    Parameters:
    -----------
    linkage_matrix : array
        Linkage matrix from hierarchical clustering
    title : str
        Plot title
    ax : matplotlib axis, optional
        Axis to plot on
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    dendrogram(linkage_matrix, ax=ax, color_threshold=0)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Data Point Index')
    ax.set_ylabel('Distance')
    ax.grid(True, alpha=0.3)


def compare_linkage_methods(X, n_clusters=3):
    """
    Compare different linkage methods side by side.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data
    n_clusters : int
        Number of clusters to extract
    """
    linkage_methods = ['single', 'complete', 'average']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    for idx, method in enumerate(linkage_methods):
        print(f"\n{'='*60}")
        print(f"Analyzing {method.upper()} linkage")
        print(f"{'='*60}")
        
        # Fit the model
        hc = HierarchicalClustering(linkage=method)
        hc.fit(X)
        
        # Get clusters
        labels = hc.get_clusters(n_clusters=n_clusters)
        
        print(f"Cluster sizes: {np.bincount(labels)}")
        
        # Plot dendrogram
        ax1 = axes[0, idx]
        plot_dendrogram(hc.get_linkage_matrix(), 
                       title=f'{method.capitalize()} Linkage - Dendrogram',
                       ax=ax1)
        
        # Plot clusters
        ax2 = axes[1, idx]
        scatter = plot_clusters(X, labels, 
                               title=f'{method.capitalize()} Linkage - Clusters',
                               ax=ax2)
        plt.colorbar(scatter, ax=ax2, label='Cluster')
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# PART 5: VALIDATION AND COMPARISON
# ============================================================================

def compare_with_sklearn(X, n_clusters=3):
    """
    Compare custom implementation with scikit-learn.
    
    Parameters:
    -----------
    X : array-like
        Input data
    n_clusters : int
        Number of clusters
    """
    print("\n" + "="*80)
    print("COMPARISON: Custom Implementation vs Scikit-Learn")
    print("="*80)
    
    linkage_methods = ['single', 'complete', 'average']
    results = []
    
    for method in linkage_methods:
        # Our implementation
        hc_custom = HierarchicalClustering(linkage=method)
        hc_custom.fit(X)
        labels_custom = hc_custom.get_clusters(n_clusters=n_clusters)
        
        # Sklearn implementation
        hc_sklearn = AgglomerativeClustering(n_clusters=n_clusters, linkage=method)
        labels_sklearn = hc_sklearn.fit_predict(X)
        
        # Calculate similarity metrics
        ari = adjusted_rand_score(labels_custom, labels_sklearn)
        nmi = normalized_mutual_info_score(labels_custom, labels_sklearn)
        
        results.append({
            'Method': method.capitalize(),
            'ARI': ari,
            'NMI': nmi,
            'Match': '✓ Perfect' if ari > 0.99 else '✗ Difference'
        })
        
        print(f"\n{method.upper()} Linkage:")
        print(f"  Adjusted Rand Index: {ari:.6f}")
        print(f"  Normalized Mutual Info: {nmi:.6f}")
        print(f"  Agreement: {results[-1]['Match']}")
    
    print("\n" + "="*80)
    print("\nNote: ARI and NMI close to 1.0 indicate high agreement.")
    print("Small differences may occur due to tie-breaking in distance calculations.")
    
    return results


def analyze_complexity(sample_sizes=[20, 50, 100, 150]):
    """
    Analyze computational complexity empirically.
    
    Parameters:
    -----------
    sample_sizes : list
        List of sample sizes to test
    """
    print("\n" + "="*70)
    print("COMPUTATIONAL COMPLEXITY ANALYSIS")
    print("="*70)
    print(f"{'N (samples)':<15} {'Time (seconds)':<20} {'Merges':<15}")
    print("="*70)
    
    timing_results = []
    
    for n in sample_sizes:
        # Generate data
        X_test, _ = make_blobs(n_samples=n, n_features=2, centers=3, random_state=42)
        
        # Measure time
        start_time = time.time()
        hc = HierarchicalClustering(linkage='single')
        hc.fit(X_test)
        elapsed_time = time.time() - start_time
        
        # Number of operations (merges)
        n_operations = len(hc.linkage_matrix_)
        
        timing_results.append((n, elapsed_time, n_operations))
        print(f"{n:<15} {elapsed_time:<20.6f} {n_operations:<15}")
    
    print("="*70)
    print("\nTheoretical Complexity: O(N³) for naive implementation")
    print("  - Distance matrix: O(N²)")
    print("  - N-1 iterations, each checking O(N²) cluster pairs")
    print("\nPossible Optimizations:")
    print("  - Priority queue: O(N² log N)")
    print("  - SLINK (single linkage): O(N²)")
    print("  - CLINK (complete linkage): O(N²)")
    
    return timing_results


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

def main():
    """
    Main demonstration of hierarchical clustering from scratch.
    """
    print("\n" + "="*80)
    print("HIERARCHICAL CLUSTERING FROM SCRATCH")
    print("Complete Implementation with Theoretical Foundations")
    print("="*80)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate sample dataset
    print("\n1. Generating sample dataset...")
    X, y_true = make_blobs(n_samples=50, n_features=2, centers=3, 
                           cluster_std=0.8, random_state=42)
    print(f"   Dataset shape: {X.shape}")
    
    # Visualize original data
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c='skyblue', s=100, edgecolors='black', alpha=0.7)
    plt.title('Sample Dataset for Hierarchical Clustering', fontsize=14, fontweight='bold')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Compare linkage methods
    print("\n2. Comparing linkage methods...")
    compare_linkage_methods(X, n_clusters=3)
    
    # Validate against sklearn
    print("\n3. Validating against scikit-learn...")
    compare_with_sklearn(X, n_clusters=3)
    
    # Analyze complexity
    print("\n4. Analyzing computational complexity...")
    timing_results = analyze_complexity([20, 50, 100, 150])
    
    # Plot complexity
    sizes = [x[0] for x in timing_results]
    times = [x[1] for x in timing_results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times, 'o-', linewidth=2, markersize=10, color='royalblue')
    plt. xlabel('Number of Samples (N)', fontsize=12)
    plt.ylabel('Execution Time (seconds)', fontsize=12)
    plt.title('Execution Time vs Dataset Size', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
