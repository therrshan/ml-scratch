import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class KMeans:
    """
    K-Means Clustering implementation from scratch
    """
    
    def __init__(self, k=3, max_iterations=100, tolerance=1e-4, init_method='random'):
        """
        Initialize K-Means clustering
        
        Parameters:
        k (int): Number of clusters
        max_iterations (int): Maximum number of iterations
        tolerance (float): Convergence tolerance
        init_method (str): Initialization method ('random' or 'kmeans++')
        """
        self.k = k
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.init_method = init_method
        self.centroids = None
        self.labels = None
        self.inertia_history = []
        self.n_iterations = 0
        
    def _euclidean_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt(np.sum((point1 - point2)**2))
    
    def _initialize_centroids(self, X):
        """Initialize centroids using specified method"""
        n_samples, n_features = X.shape
        
        if self.init_method == 'random':
            # Random initialization within data bounds
            min_vals = np.min(X, axis=0)
            max_vals = np.max(X, axis=0)
            self.centroids = np.random.uniform(min_vals, max_vals, (self.k, n_features))
            
        elif self.init_method == 'kmeans++':
            # K-means++ initialization
            self.centroids = np.zeros((self.k, n_features))
            
            # Choose first centroid randomly
            self.centroids[0] = X[np.random.randint(n_samples)]
            
            # Choose remaining centroids
            for i in range(1, self.k):
                distances = np.zeros(n_samples)
                
                for j, point in enumerate(X):
                    # Find distance to nearest existing centroid
                    min_dist = float('inf')
                    for existing_centroid in self.centroids[:i]:
                        dist = self._euclidean_distance(point, existing_centroid)
                        min_dist = min(min_dist, dist)
                    distances[j] = min_dist**2
                
                # Choose next centroid with probability proportional to squared distance
                probabilities = distances / np.sum(distances)
                cumulative_probs = np.cumsum(probabilities)
                r = np.random.random()
                
                for j, cumulative_prob in enumerate(cumulative_probs):
                    if r <= cumulative_prob:
                        self.centroids[i] = X[j]
                        break
        else:
            raise ValueError(f"Unknown initialization method: {self.init_method}")
    
    def _assign_clusters(self, X):
        """Assign each point to the nearest centroid"""
        n_samples = X.shape[0]
        labels = np.zeros(n_samples, dtype=int)
        
        for i, point in enumerate(X):
            distances = [self._euclidean_distance(point, centroid) 
                        for centroid in self.centroids]
            labels[i] = np.argmin(distances)
        
        return labels
    
    def _update_centroids(self, X, labels):
        """Update centroids based on current cluster assignments"""
        new_centroids = np.zeros_like(self.centroids)
        
        for k in range(self.k):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                new_centroids[k] = np.mean(cluster_points, axis=0)
            else:
                # If cluster is empty, keep old centroid
                new_centroids[k] = self.centroids[k]
        
        return new_centroids
    
    def _calculate_inertia(self, X, labels):
        """Calculate within-cluster sum of squares (inertia)"""
        inertia = 0
        for k in range(self.k):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                centroid = self.centroids[k]
                inertia += np.sum([self._euclidean_distance(point, centroid)**2 
                                 for point in cluster_points])
        return inertia
    
    def fit(self, X):
        """
        Fit K-means clustering to data
        
        Parameters:
        X (array): Data to cluster (m x n)
        """
        X = np.array(X)
        n_samples, n_features = X.shape
        
        if self.k > n_samples:
            raise ValueError(f"Number of clusters ({self.k}) cannot exceed number of samples ({n_samples})")
        
        # Initialize centroids
        self._initialize_centroids(X)
        
        # Main K-means loop
        for iteration in range(self.max_iterations):
            # Assign points to clusters
            labels = self._assign_clusters(X)
            
            # Calculate inertia
            inertia = self._calculate_inertia(X, labels)
            self.inertia_history.append(inertia)
            
            # Update centroids
            new_centroids = self._update_centroids(X, labels)
            
            # Check for convergence
            centroid_shift = np.mean([self._euclidean_distance(old, new) 
                                    for old, new in zip(self.centroids, new_centroids)])
            
            self.centroids = new_centroids
            self.n_iterations = iteration + 1
            
            if centroid_shift < self.tolerance:
                print(f"Converged after {self.n_iterations} iterations")
                break
        else:
            print(f"Reached maximum iterations ({self.max_iterations})")
        
        # Final assignment
        self.labels = self._assign_clusters(X)
        
        print(f"K-Means clustering completed with inertia: {self.inertia_history[-1]:.4f}")
    
    def predict(self, X):
        """
        Predict cluster labels for new data
        
        Parameters:
        X (array): Data to predict clusters for
        
        Returns:
        array: Predicted cluster labels
        """
        if self.centroids is None:
            raise ValueError("Model must be fitted before making predictions")
        
        X = np.array(X)
        return self._assign_clusters(X)
    
    def fit_predict(self, X):
        """
        Fit the model and return cluster labels
        
        Parameters:
        X (array): Data to cluster
        
        Returns:
        array: Cluster labels
        """
        self.fit(X)
        return self.labels
    
    def get_cluster_centers(self):
        """Get cluster centroids"""
        return self.centroids.copy() if self.centroids is not None else None
    
    def plot_clusters(self, X, show_centroids=True, title="K-Means Clustering"):
        """
        Plot clusters for 2D data
        
        Parameters:
        X (array): 2D data array
        show_centroids (bool): Whether to show centroids
        title (str): Plot title
        """
        if X.shape[1] != 2:
            print("Cluster plot only available for 2D data")
            return
        
        if self.labels is None:
            print("Model must be fitted first")
            return
        
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
        
        plt.figure(figsize=(10, 8))
        
        # Plot data points
        for k in range(self.k):
            cluster_points = X[self.labels == k]
            if len(cluster_points) > 0:
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                          c=colors[k % len(colors)], alpha=0.7, s=50, 
                          label=f'Cluster {k}')
        
        # Plot centroids
        if show_centroids and self.centroids is not None:
            plt.scatter(self.centroids[:, 0], self.centroids[:, 1], 
                       c='black', marker='x', s=200, linewidths=3, 
                       label='Centroids')
        
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_inertia_history(self):
        """Plot inertia over iterations"""
        if not self.inertia_history:
            print("No inertia history available")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.inertia_history) + 1), self.inertia_history, 'bo-')
        plt.xlabel('Iteration')
        plt.ylabel('Inertia (Within-cluster sum of squares)')
        plt.title('K-Means Convergence')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def elbow_method(self, X, k_range=range(1, 11), n_runs=5):
        """
        Use elbow method to find optimal number of clusters
        
        Parameters:
        X (array): Data to cluster
        k_range (range): Range of k values to test
        n_runs (int): Number of runs per k value (for stability)
        
        Returns:
        dict: Results with k values and corresponding inertias
        """
        results = {'k_values': [], 'inertias': [], 'inertias_std': []}
        
        for k in k_range:
            inertias = []
            
            # Run multiple times for each k to get stable results
            for _ in range(n_runs):
                kmeans_temp = KMeans(k=k, init_method=self.init_method)
                kmeans_temp.fit(X)
                inertias.append(kmeans_temp.inertia_history[-1])
            
            mean_inertia = np.mean(inertias)
            std_inertia = np.std(inertias)
            
            results['k_values'].append(k)
            results['inertias'].append(mean_inertia)
            results['inertias_std'].append(std_inertia)
            
            print(f"k={k}: Inertia = {mean_inertia:.2f} ± {std_inertia:.2f}")
        
        # Plot elbow curve
        plt.figure(figsize=(10, 6))
        plt.errorbar(results['k_values'], results['inertias'], 
                    yerr=results['inertias_std'], fmt='bo-', capsize=5)
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method for Optimal k')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return results
    
    def silhouette_analysis(self, X):
        """
        Calculate silhouette score for current clustering
        
        Parameters:
        X (array): Data that was clustered
        
        Returns:
        float: Average silhouette score
        """
        if self.labels is None:
            raise ValueError("Model must be fitted first")
        
        n_samples = len(X)
        silhouette_scores = []
        
        for i in range(n_samples):
            # Current point and its cluster
            point = X[i]
            cluster_label = self.labels[i]
            
            # Calculate a(i): mean distance to other points in same cluster
            same_cluster_points = X[self.labels == cluster_label]
            if len(same_cluster_points) == 1:
                a_i = 0  # Only point in cluster
            else:
                a_i = np.mean([self._euclidean_distance(point, other_point) 
                              for other_point in same_cluster_points 
                              if not np.array_equal(point, other_point)])
            
            # Calculate b(i): mean distance to points in nearest other cluster
            b_i = float('inf')
            for other_cluster in range(self.k):
                if other_cluster != cluster_label:
                    other_cluster_points = X[self.labels == other_cluster]
                    if len(other_cluster_points) > 0:
                        mean_dist = np.mean([self._euclidean_distance(point, other_point) 
                                           for other_point in other_cluster_points])
                        b_i = min(b_i, mean_dist)
            
            # Calculate silhouette score for this point
            if max(a_i, b_i) == 0:
                s_i = 0
            else:
                s_i = (b_i - a_i) / max(a_i, b_i)
            
            silhouette_scores.append(s_i)
        
        return np.mean(silhouette_scores)
    
    def plot_cluster_evolution(self, X, save_steps=True):
        """
        Plot how clusters evolve during training
        
        Parameters:
        X (array): 2D data to cluster
        save_steps (bool): Whether to save intermediate steps
        """
        if X.shape[1] != 2:
            print("Evolution plot only available for 2D data")
            return
        
        # Store original state
        original_centroids = self.centroids.copy() if self.centroids is not None else None
        original_labels = self.labels.copy() if self.labels is not None else None
        
        # Re-initialize and track evolution
        self._initialize_centroids(X)
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
        
        evolution_data = []
        
        for iteration in range(min(6, self.max_iterations)):  # Show first 6 iterations
            # Assign points to clusters
            labels = self._assign_clusters(X)
            
            # Store current state
            evolution_data.append({
                'centroids': self.centroids.copy(),
                'labels': labels.copy()
            })
            
            # Update centroids
            new_centroids = self._update_centroids(X, labels)
            
            # Check convergence
            centroid_shift = np.mean([self._euclidean_distance(old, new) 
                                    for old, new in zip(self.centroids, new_centroids)])
            
            self.centroids = new_centroids
            
            if centroid_shift < self.tolerance:
                break
        
        # Plot evolution
        n_steps = len(evolution_data)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for i, data in enumerate(evolution_data):
            if i >= 6:
                break
                
            ax = axes[i]
            
            # Plot data points colored by cluster
            for k in range(self.k):
                cluster_points = X[data['labels'] == k]
                if len(cluster_points) > 0:
                    ax.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                             c=colors[k % len(colors)], alpha=0.7, s=30)
            
            # Plot centroids
            ax.scatter(data['centroids'][:, 0], data['centroids'][:, 1], 
                      c='black', marker='x', s=200, linewidths=3)
            
            ax.set_title(f'Iteration {i}')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_steps, 6):
            axes[i].axis('off')
        
        plt.suptitle('K-Means Clustering Evolution', fontsize=16)
        plt.tight_layout()
        plt.show()
        
        # Restore original state
        self.centroids = original_centroids
        self.labels = original_labels

# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    
    # Create 3 clusters of data
    cluster1 = np.random.normal([2, 2], 0.5, (50, 2))
    cluster2 = np.random.normal([6, 6], 0.5, (50, 2))
    cluster3 = np.random.normal([2, 6], 0.5, (50, 2))
    
    X = np.vstack([cluster1, cluster2, cluster3])
    
    print("=== K-Means Clustering Example ===")
    
    # Create and fit K-means model
    kmeans = KMeans(k=3, init_method='kmeans++')
    labels = kmeans.fit_predict(X)
    
    print(f"Final inertia: {kmeans.inertia_history[-1]:.4f}")
    print(f"Number of iterations: {kmeans.n_iterations}")
    
    # Calculate silhouette score
    silhouette_score = kmeans.silhouette_analysis(X)
    print(f"Silhouette score: {silhouette_score:.4f}")
    
    # Plot results
    kmeans.plot_clusters(X, title="K-Means Clustering Results")
    kmeans.plot_inertia_history()
    
    # Show cluster evolution
    print("\n=== Cluster Evolution Visualization ===")
    kmeans_evolution = KMeans(k=3, init_method='random', max_iterations=10)
    kmeans_evolution.plot_cluster_evolution(X)
    
    # Find optimal k using elbow method
    print("\n=== Elbow Method Analysis ===")
    elbow_results = kmeans.elbow_method(X, k_range=range(1, 8), n_runs=3)
    
    # Compare different initialization methods
    print("\n=== Comparing Initialization Methods ===")
    methods = ['random', 'kmeans++']
    
    plt.figure(figsize=(15, 5))
    
    for i, method in enumerate(methods):
        kmeans_comp = KMeans(k=3, init_method=method, max_iterations=100)
        kmeans_comp.fit(X)
        
        plt.subplot(1, 2, i+1)
        
        colors = ['red', 'blue', 'green']
        for k in range(3):
            cluster_points = X[kmeans_comp.labels == k]
            if len(cluster_points) > 0:
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                          c=colors[k], alpha=0.7, s=50, label=f'Cluster {k}')
        
        plt.scatter(kmeans_comp.centroids[:, 0], kmeans_comp.centroids[:, 1], 
                   c='black', marker='x', s=200, linewidths=3, label='Centroids')
        
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(f'K-Means with {method} initialization\n'
                 f'Inertia: {kmeans_comp.inertia_history[-1]:.2f}, '
                 f'Iterations: {kmeans_comp.n_iterations}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Test prediction on new data
    print("\n=== Prediction on New Data ===")
    new_points = np.array([[1, 1], [7, 7], [3, 5]])
    predictions = kmeans.predict(new_points)
    
    for i, (point, pred) in enumerate(zip(new_points, predictions)):
        print(f"Point {point} predicted to be in cluster {pred}")
    
    # Visualize predictions
    plt.figure(figsize=(10, 8))
    
    # Plot original clusters
    colors = ['red', 'blue', 'green']
    for k in range(3):
        cluster_points = X[kmeans.labels == k]
        if len(cluster_points) > 0:
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                      c=colors[k], alpha=0.7, s=50, label=f'Cluster {k}')
    
    # Plot centroids
    plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], 
               c='black', marker='x', s=200, linewidths=3, label='Centroids')
    
    # Plot new points
    for i, (point, pred) in enumerate(zip(new_points, predictions)):
        plt.scatter(point[0], point[1], c=colors[pred], marker='s', 
                   s=150, edgecolors='black', linewidth=3, 
                   label=f'New point {i}' if i == 0 else "")
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('K-Means Clustering with New Point Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("\n=== Different Data Distributions ===")
    
    # Test on different data distributions
    datasets = {
        'Blobs': X,
        'Circles': None,
        'Moons': None
    }
    
    # Generate circles data
    np.random.seed(42)
    angles = np.linspace(0, 2*np.pi, 100)
    inner_circle = np.column_stack([0.5 * np.cos(angles), 0.5 * np.sin(angles)])
    outer_circle = np.column_stack([2 * np.cos(angles), 2 * np.sin(angles)])
    datasets['Circles'] = np.vstack([inner_circle, outer_circle])
    
    # Generate moons data (simplified)
    t = np.linspace(0, np.pi, 50)
    moon1 = np.column_stack([np.cos(t), np.sin(t)])
    moon2 = np.column_stack([1 - np.cos(t), 1 - np.sin(t) - 0.5])
    datasets['Moons'] = np.vstack([moon1, moon2])
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, (name, data) in enumerate(datasets.items()):
        if data is None:
            continue
            
        # Fit K-means
        k_val = 3 if name == 'Blobs' else 2
        kmeans_test = KMeans(k=k_val, init_method='kmeans++')
        kmeans_test.fit(data)
        
        ax = axes[i]
        
        # Plot clusters
        colors = ['red', 'blue', 'green']
        for k in range(k_val):
            cluster_points = data[kmeans_test.labels == k]
            if len(cluster_points) > 0:
                ax.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                         c=colors[k], alpha=0.7, s=30, label=f'Cluster {k}')
        
        # Plot centroids
        ax.scatter(kmeans_test.centroids[:, 0], kmeans_test.centroids[:, 1], 
                  c='black', marker='x', s=200, linewidths=3, label='Centroids')
        
        silhouette = kmeans_test.silhouette_analysis(data)
        ax.set_title(f'{name} Dataset\nSilhouette Score: {silhouette:.3f}')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()