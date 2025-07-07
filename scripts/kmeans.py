"""
K‑Means clustering from scratch with random or k‑means++ initialization plus inertia, silhouette,
elbow diagnostics, and rich visualization utilities.
"""

import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, k=3, max_iterations=100, tolerance=1e-4, init_method='random'):
        self.k = k
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.init_method = init_method
        self.centroids = None
        self.labels = None
        self.inertia_history = []
        self.n_iterations = 0

    def _euclidean_distance(self, point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def _initialize_centroids(self, X):
        n_samples, n_features = X.shape
        if self.init_method == 'random':
            min_vals = np.min(X, axis=0)
            max_vals = np.max(X, axis=0)
            self.centroids = np.random.uniform(min_vals, max_vals, (self.k, n_features))
        elif self.init_method == 'kmeans++':
            self.centroids = np.zeros((self.k, n_features))
            self.centroids[0] = X[np.random.randint(n_samples)]
            for i in range(1, self.k):
                distances = np.zeros(n_samples)
                for j, point in enumerate(X):
                    min_dist = float('inf')
                    for existing_centroid in self.centroids[:i]:
                        dist = self._euclidean_distance(point, existing_centroid)
                        min_dist = min(min_dist, dist)
                    distances[j] = min_dist ** 2
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
        n_samples = X.shape[0]
        labels = np.zeros(n_samples, dtype=int)
        for i, point in enumerate(X):
            distances = [self._euclidean_distance(point, centroid) for centroid in self.centroids]
            labels[i] = np.argmin(distances)
        return labels

    def _update_centroids(self, X, labels):
        new_centroids = np.zeros_like(self.centroids)
        for k in range(self.k):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                new_centroids[k] = np.mean(cluster_points, axis=0)
            else:
                new_centroids[k] = self.centroids[k]
        return new_centroids

    def _calculate_inertia(self, X, labels):
        inertia = 0
        for k in range(self.k):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                centroid = self.centroids[k]
                inertia += np.sum([(self._euclidean_distance(point, centroid) ** 2) for point in cluster_points])
        return inertia

    def fit(self, X):
        X = np.array(X)
        n_samples, _ = X.shape
        if self.k > n_samples:
            raise ValueError(f"Number of clusters ({self.k}) cannot exceed number of samples ({n_samples})")
        self._initialize_centroids(X)
        for iteration in range(self.max_iterations):
            labels = self._assign_clusters(X)
            inertia = self._calculate_inertia(X, labels)
            self.inertia_history.append(inertia)
            new_centroids = self._update_centroids(X, labels)
            centroid_shift = np.mean([self._euclidean_distance(old, new) for old, new in zip(self.centroids, new_centroids)])
            self.centroids = new_centroids
            self.n_iterations = iteration + 1
            if centroid_shift < self.tolerance:
                break
        self.labels = self._assign_clusters(X)

    def predict(self, X):
        if self.centroids is None:
            raise ValueError("Model must be fitted before making predictions")
        X = np.array(X)
        return self._assign_clusters(X)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels

    def get_cluster_centers(self):
        return self.centroids.copy() if self.centroids is not None else None

    def plot_clusters(self, X, show_centroids=True, title="K-Means Clustering"):
        if X.shape[1] != 2 or self.labels is None:
            return
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
        plt.figure(figsize=(10, 8))
        for k in range(self.k):
            cluster_points = X[self.labels == k]
            if len(cluster_points) > 0:
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[k % len(colors)], alpha=0.7, s=50, label=f'Cluster {k}')
        if show_centroids and self.centroids is not None:
            plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='black', marker='x', s=200, linewidths=3, label='Centroids')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_inertia_history(self):
        if not self.inertia_history:
            return
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.inertia_history) + 1), self.inertia_history, 'bo-')
        plt.xlabel('Iteration')
        plt.ylabel('Inertia (Within-cluster sum of squares)')
        plt.title('K-Means Convergence')
        plt.grid(True, alpha=0.3)
        plt.show()

    def elbow_method(self, X, k_range=range(1, 11), n_runs=5):
        results = {'k_values': [], 'inertias': [], 'inertias_std': []}
        for k in k_range:
            inertias = []
            for _ in range(n_runs):
                kmeans_temp = KMeans(k=k, init_method=self.init_method)
                kmeans_temp.fit(X)
                inertias.append(kmeans_temp.inertia_history[-1])
            results['k_values'].append(k)
            results['inertias'].append(np.mean(inertias))
            results['inertias_std'].append(np.std(inertias))
        plt.figure(figsize=(10, 6))
        plt.errorbar(results['k_values'], results['inertias'], yerr=results['inertias_std'], fmt='bo-', capsize=5)
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method for Optimal k')
        plt.grid(True, alpha=0.3)
        plt.show()
        return results

    def silhouette_analysis(self, X):
        if self.labels is None:
            raise ValueError("Model must be fitted first")
        n_samples = len(X)
        silhouette_scores = []
        for i in range(n_samples):
            point = X[i]
            cluster_label = self.labels[i]
            same_cluster_points = X[self.labels == cluster_label]
            if len(same_cluster_points) == 1:
                a_i = 0
            else:
                a_i = np.mean([self._euclidean_distance(point, other_point) for other_point in same_cluster_points if not np.array_equal(point, other_point)])
            b_i = float('inf')
            for other_cluster in range(self.k):
                if other_cluster != cluster_label:
                    other_cluster_points = X[self.labels == other_cluster]
                    if len(other_cluster_points) > 0:
                        mean_dist = np.mean([self._euclidean_distance(point, other_point) for other_point in other_cluster_points])
                        b_i = min(b_i, mean_dist)
            s_i = 0 if max(a_i, b_i) == 0 else (b_i - a_i) / max(a_i, b_i)
            silhouette_scores.append(s_i)
        return np.mean(silhouette_scores)

    def plot_cluster_evolution(self, X):
        if X.shape[1] != 2:
            return
        original_centroids = self.centroids.copy() if self.centroids is not None else None
        original_labels = self.labels.copy() if self.labels is not None else None
        self._initialize_centroids(X)
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
        evolution_data = []
        for iteration in range(min(6, self.max_iterations)):
            labels = self._assign_clusters(X)
            evolution_data.append({'centroids': self.centroids.copy(), 'labels': labels.copy()})
            new_centroids = self._update_centroids(X, labels)
            centroid_shift = np.mean([self._euclidean_distance(old, new) for old, new in zip(self.centroids, new_centroids)])
            self.centroids = new_centroids
            if centroid_shift < self.tolerance:
                break
        n_steps = len(evolution_data)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        for i, data in enumerate(evolution_data):
            if i >= 6:
                break
            ax = axes[i]
            for k in range(self.k):
                cluster_points = X[data['labels'] == k]
                if len(cluster_points) > 0:
                    ax.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[k % len(colors)], alpha=0.7, s=30)
            ax.scatter(data['centroids'][:, 0], data['centroids'][:, 1], c='black', marker='x', s=200, linewidths=3)
            ax.set_title(f'Iteration {i}')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.grid(True, alpha=0.3)
        for i in range(n_steps, 6):
            axes[i].axis('off')
        plt.suptitle('K-Means Clustering Evolution', fontsize=16)
        plt.tight_layout()
        plt.show()
        self.centroids = original_centroids
        self.labels = original_labels

if __name__ == "__main__":

    np.random.seed(42)
    cluster1 = np.random.normal([2, 2], 0.5, (50, 2))
    cluster2 = np.random.normal([6, 6], 0.5, (50, 2))
    cluster3 = np.random.normal([2, 6], 0.5, (50, 2))
    
    X = np.vstack([cluster1, cluster2, cluster3])
    
    print("=== K-Means Clustering Example ===")

    kmeans = KMeans(k=3, init_method='kmeans++')
    labels = kmeans.fit_predict(X)
    
    print(f"Final inertia: {kmeans.inertia_history[-1]:.4f}")
    print(f"Number of iterations: {kmeans.n_iterations}")

    silhouette_score = kmeans.silhouette_analysis(X)
    print(f"Silhouette score: {silhouette_score:.4f}")

    kmeans.plot_clusters(X, title="K-Means Clustering Results")
    kmeans.plot_inertia_history()

    print("\n=== Cluster Evolution Visualization ===")
    kmeans_evolution = KMeans(k=3, init_method='random', max_iterations=10)
    kmeans_evolution.plot_cluster_evolution(X)

    print("\n=== Elbow Method Analysis ===")
    elbow_results = kmeans.elbow_method(X, k_range=range(1, 8), n_runs=3)

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

    print("\n=== Prediction on New Data ===")
    new_points = np.array([[1, 1], [7, 7], [3, 5]])
    predictions = kmeans.predict(new_points)
    
    for i, (point, pred) in enumerate(zip(new_points, predictions)):
        print(f"Point {point} predicted to be in cluster {pred}")

    plt.figure(figsize=(10, 8))

    colors = ['red', 'blue', 'green']
    for k in range(3):
        cluster_points = X[kmeans.labels == k]
        if len(cluster_points) > 0:
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                      c=colors[k], alpha=0.7, s=50, label=f'Cluster {k}')

    plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], 
               c='black', marker='x', s=200, linewidths=3, label='Centroids')

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

    datasets = {
        'Blobs': X,
        'Circles': None,
        'Moons': None
    }

    np.random.seed(42)
    angles = np.linspace(0, 2*np.pi, 100)
    inner_circle = np.column_stack([0.5 * np.cos(angles), 0.5 * np.sin(angles)])
    outer_circle = np.column_stack([2 * np.cos(angles), 2 * np.sin(angles)])
    datasets['Circles'] = np.vstack([inner_circle, outer_circle])

    t = np.linspace(0, np.pi, 50)
    moon1 = np.column_stack([np.cos(t), np.sin(t)])
    moon2 = np.column_stack([1 - np.cos(t), 1 - np.sin(t) - 0.5])
    datasets['Moons'] = np.vstack([moon1, moon2])
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, (name, data) in enumerate(datasets.items()):
        if data is None:
            continue

        k_val = 3 if name == 'Blobs' else 2
        kmeans_test = KMeans(k=k_val, init_method='kmeans++')
        kmeans_test.fit(data)
        
        ax = axes[i]

        colors = ['red', 'blue', 'green']
        for k in range(k_val):
            cluster_points = data[kmeans_test.labels == k]
            if len(cluster_points) > 0:
                ax.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                         c=colors[k], alpha=0.7, s=30, label=f'Cluster {k}')

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
