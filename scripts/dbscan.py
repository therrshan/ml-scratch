"""
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) implemented
from scratch with batch processing, visualization, and parameter optimization.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import silhouette_score
from collections import Counter
import pandas as pd

class DBSCAN:
    
    def __init__(self, eps, min_pts, batch_size=1000, metric='euclidean'):
        self.eps = eps
        self.min_pts = min_pts
        self.labels = None
        self.batch_size = batch_size
        self.metric = metric
        self.core_points = []
        self.border_points = []
        self.noise_points = []
        
    def _euclidean_distance_batch(self, batch, coords):
        dists = np.sum(batch**2, axis=1, keepdims=True) + np.sum(coords**2, axis=1) - 2 * np.dot(batch, coords.T)
        return np.sqrt(np.maximum(dists, 0))
    
    def _manhattan_distance_batch(self, batch, coords):
        dists = np.zeros((len(batch), len(coords)))
        for i, point in enumerate(batch):
            dists[i] = np.sum(np.abs(coords - point), axis=1)
        return dists
    
    def find_neighbors(self, df):
        if isinstance(df, pd.DataFrame):
            coords = df.values
        else:
            coords = df
            
        num_points = len(coords)
        neighbors = [[] for _ in range(num_points)]
        
        print(f"Computing neighbors for {num_points} points with batch size={self.batch_size}...")
        
        for start in tqdm(range(0, num_points, self.batch_size), desc="Processing batches"):
            end = min(start + self.batch_size, num_points)
            batch = coords[start:end]
            
            if self.metric == 'euclidean':
                dists = self._euclidean_distance_batch(batch, coords)
            elif self.metric == 'manhattan':
                dists = self._manhattan_distance_batch(batch, coords)
            else:
                raise ValueError(f"Unknown metric: {self.metric}")
            
            for i, row in enumerate(dists):
                neighbors[start + i] = np.where((row <= self.eps) & (row > 0))[0].tolist()
        
        print("Finished computing neighbors.")
        return neighbors
    
    def dbscan(self, df):
        if isinstance(df, pd.DataFrame):
            data = df.values
        else:
            data = df
            
        print(f"\nRunning DBSCAN with eps={self.eps} and min_pts={self.min_pts}...")
        neighbors = self.find_neighbors(data)
        self.labels = [0] * len(data)
        C = 0
        
        self.core_points = []
        for p in range(len(data)):
            if len(neighbors[p]) >= self.min_pts:
                self.core_points.append(p)
        
        print(f"Found {len(self.core_points)} core points")
        
        for p in tqdm(self.core_points, desc='Processing core points'):
            if self.labels[p] != 0:
                continue
            C += 1
            self.grow_cluster(p, neighbors, C)
        
        self.noise_points = [i for i, label in enumerate(self.labels) if label == -1]
        self.border_points = [i for i, label in enumerate(self.labels) 
                             if label > 0 and i not in self.core_points]
        
        print(f"DBSCAN complete. Found {C} clusters.")
        print(f"Core points: {len(self.core_points)}, Border points: {len(self.border_points)}, "
              f"Noise points: {len(self.noise_points)}")
        
        return self.labels
    
    def grow_cluster(self, p, neighbors, C):
        self.labels[p] = C
        queue = neighbors[p].copy()
        
        while queue:
            pn = queue.pop(0)
            
            if self.labels[pn] == -1:
                self.labels[pn] = C
            elif self.labels[pn] == 0:
                self.labels[pn] = C
                if len(neighbors[pn]) >= self.min_pts:
                    queue.extend([n for n in neighbors[pn] if self.labels[n] == 0])
    
    def fit(self, X):
        self.labels_ = np.array(self.dbscan(X))
        return self
    
    def fit_predict(self, X):
        self.fit(X)
        return self.labels_
    
    def plot_results(self, X, title="DBSCAN Clustering Results"):
        if X.shape[1] != 2:
            print("Plotting only available for 2D data")
            return
        
        plt.figure(figsize=(12, 8))
        
        unique_labels = set(self.labels_)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            if label == -1:
                color = 'black'
                marker_size = 6
            else:
                marker_size = 12
            
            mask = self.labels_ == label
            plt.scatter(X[mask, 0], X[mask, 1], c=[color], s=marker_size, 
                       label=f'Cluster {label}' if label != -1 else 'Noise',
                       alpha=0.8 if label != -1 else 0.4,
                       edgecolors='black' if label != -1 else 'none',
                       linewidths=0.5)
        
        # if hasattr(self, 'core_points') and len(self.core_points) > 0:
        #     core_mask = np.zeros(len(X), dtype=bool)
        #     core_mask[self.core_points] = True
        #     plt.scatter(X[core_mask, 0], X[core_mask, 1], 
        #                marker='+', s=200, c='red', label='Core points')
        
        plt.title(title)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_eps_analysis(self, X, k=4):
        from sklearn.neighbors import NearestNeighbors
        
        neighbors = NearestNeighbors(n_neighbors=k+1)
        neighbors_fit = neighbors.fit(X)
        distances, indices = neighbors_fit.kneighbors(X)
        
        k_distances = np.sort(distances[:, k], axis=0)
        
        plt.figure(figsize=(10, 6))
        plt.plot(k_distances)
        plt.xlabel('Points sorted by distance')
        plt.ylabel(f'{k}-NN Distance')
        plt.title(f'K-distance Graph (k={k}) for Epsilon Selection')
        plt.grid(True, alpha=0.3)
        
        gradient = np.gradient(k_distances)
        knee_point = np.argmax(gradient)
        suggested_eps = k_distances[knee_point]
        
        plt.axhline(y=suggested_eps, color='r', linestyle='--', 
                   label=f'Suggested eps: {suggested_eps:.3f}')
        plt.legend()
        plt.show()
        
        return suggested_eps


def evaluate_and_visualize(dbscan, X, dataset_name):
    labels = dbscan.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = sum(labels == -1)
    
    cluster_sizes = Counter(labels)
    if -1 in cluster_sizes:
        del cluster_sizes[-1]
    
    print(f"\nDataset: {dataset_name}")
    print(f"Number of clusters: {n_clusters}")
    print(f"Number of noise points: {n_noise} ({100*n_noise/len(labels):.1f}%)")
    print(f"Cluster sizes: {dict(cluster_sizes)}")
    
    non_noise_mask = labels != -1
    if n_clusters > 1 and non_noise_mask.sum() > 1:
        sil_score = silhouette_score(X[non_noise_mask], labels[non_noise_mask])
        print(f"Silhouette Score (excluding noise): {sil_score:.4f}")
    else:
        print("Silhouette Score: Not applicable")
    
    return n_clusters, n_noise, cluster_sizes


def parameter_search(X, eps_range, min_pts_range):
    results = []
    
    print("Performing parameter search...")
    for eps in eps_range:
        for min_pts in min_pts_range:
            dbscan = DBSCAN(eps=eps, min_pts=min_pts)
            labels = dbscan.fit_predict(X)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = sum(labels == -1)
            
            if n_clusters > 1 and n_noise < len(X):
                non_noise_mask = labels != -1
                if non_noise_mask.sum() > n_clusters:
                    sil_score = silhouette_score(X[non_noise_mask], labels[non_noise_mask])
                else:
                    sil_score = -1
            else:
                sil_score = -1
            
            results.append({
                'eps': eps,
                'min_pts': min_pts,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'silhouette': sil_score
            })
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    print("=== DBSCAN Clustering Example ===")
    
    np.random.seed(42)
    
    print("\n=== Generating Sample Datasets ===")
    
    from sklearn.datasets import make_moons, make_circles, make_blobs
    
    n_samples = 750
    X_moons, y_moons = make_moons(n_samples=n_samples, noise=0.05, random_state=42)
    X_circles, y_circles = make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=42)
    X_blobs, y_blobs = make_blobs(n_samples=n_samples, centers=3, random_state=42)
    
    noise = np.random.uniform(-6, 6, (50, 2))
    X_blobs = np.vstack([X_blobs, noise])
    y_blobs = np.hstack([y_blobs, [-1] * 50])
    
    datasets = [
        ("Moons", X_moons),
        ("Circles", X_circles),
        ("Blobs with Noise", X_blobs)
    ]
    
    print("\n=== K-Distance Analysis for Parameter Selection ===")
    
    for name, X in datasets[:1]:
        print(f"\nAnalyzing {name} dataset...")
        dbscan_temp = DBSCAN(eps=0.1, min_pts=5)
        suggested_eps = dbscan_temp.plot_eps_analysis(X, k=5)
        print(f"Suggested epsilon: {suggested_eps:.3f}")
    
    print("\n=== Running DBSCAN on Different Datasets ===")
    
    parameters = {
        "Moons": (0.2, 5),
        "Circles": (0.15, 5),
        "Blobs with Noise": (0.5, 5)
    }
    
    for name, X in datasets:
        print(f"\n{'='*50}")
        print(f"Dataset: {name}")
        print(f"{'='*50}")
        
        eps, min_pts = parameters[name]
        dbscan = DBSCAN(eps=eps, min_pts=min_pts)
        dbscan.fit(X)
        
        evaluate_and_visualize(dbscan, X, name)
        dbscan.plot_results(X, title=f'DBSCAN on {name} Dataset')
    
    print("\n=== Comparing Different Distance Metrics ===")
    
    X_test = X_moons
    metrics = ['euclidean', 'manhattan']
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    for i, metric in enumerate(metrics):
        dbscan = DBSCAN(eps=0.2, min_pts=5, metric=metric)
        dbscan.fit(X_test)
        
        ax = axes[i]
        unique_labels = set(dbscan.labels_)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            if label == -1:
                color = 'black'
            mask = dbscan.labels_ == label
            ax.scatter(X_test[mask, 0], X_test[mask, 1], c=[color], s=12,
                      label=f'Cluster {label}' if label != -1 else 'Noise',
                      alpha=0.8 if label != -1 else 0.4)
        
        ax.set_title(f'DBSCAN with {metric.title()} Distance')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== Parameter Sensitivity Analysis ===")
    
    eps_range = np.linspace(0.1, 0.5, 9)
    min_pts_range = range(3, 10)
    
    results_df = parameter_search(X_moons, eps_range, min_pts_range)
    
    best_params = results_df.loc[results_df['silhouette'].idxmax()]
    print(f"\nBest parameters for Moons dataset:")
    print(f"eps: {best_params['eps']:.3f}, min_pts: {int(best_params['min_pts'])}")
    print(f"Clusters: {int(best_params['n_clusters'])}, Silhouette: {best_params['silhouette']:.4f}")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    pivot_clusters = results_df.pivot(index='min_pts', columns='eps', values='n_clusters')
    im1 = axes[0, 0].imshow(pivot_clusters, cmap='viridis', aspect='auto')
    axes[0, 0].set_title('Number of Clusters')
    axes[0, 0].set_xlabel('eps')
    axes[0, 0].set_ylabel('min_pts')
    axes[0, 0].set_xticks(range(len(eps_range)))
    axes[0, 0].set_xticklabels([f'{e:.2f}' for e in eps_range])
    axes[0, 0].set_yticks(range(len(min_pts_range)))
    axes[0, 0].set_yticklabels(min_pts_range)
    plt.colorbar(im1, ax=axes[0, 0])
    
    pivot_noise = results_df.pivot(index='min_pts', columns='eps', values='n_noise')
    im2 = axes[0, 1].imshow(pivot_noise, cmap='Reds', aspect='auto')
    axes[0, 1].set_title('Number of Noise Points')
    axes[0, 1].set_xlabel('eps')
    axes[0, 1].set_ylabel('min_pts')
    axes[0, 1].set_xticks(range(len(eps_range)))
    axes[0, 1].set_xticklabels([f'{e:.2f}' for e in eps_range])
    axes[0, 1].set_yticks(range(len(min_pts_range)))
    axes[0, 1].set_yticklabels(min_pts_range)
    plt.colorbar(im2, ax=axes[0, 1])
    
    pivot_silhouette = results_df.pivot(index='min_pts', columns='eps', values='silhouette')
    im3 = axes[1, 0].imshow(pivot_silhouette, cmap='coolwarm', aspect='auto')
    axes[1, 0].set_title('Silhouette Score')
    axes[1, 0].set_xlabel('eps')
    axes[1, 0].set_ylabel('min_pts')
    axes[1, 0].set_xticks(range(len(eps_range)))
    axes[1, 0].set_xticklabels([f'{e:.2f}' for e in eps_range])
    axes[1, 0].set_yticks(range(len(min_pts_range)))
    axes[1, 0].set_yticklabels(min_pts_range)
    plt.colorbar(im3, ax=axes[1, 0])
    
    best_idx = results_df['silhouette'].idxmax()
    best_eps = results_df.loc[best_idx, 'eps']
    best_min_pts = results_df.loc[best_idx, 'min_pts']
    
    dbscan_best = DBSCAN(eps=best_eps, min_pts=int(best_min_pts))
    dbscan_best.fit(X_moons)
    
    ax = axes[1, 1]
    unique_labels = set(dbscan_best.labels_)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        if label == -1:
            color = 'black'
        mask = dbscan_best.labels_ == label
        ax.scatter(X_moons[mask, 0], X_moons[mask, 1], c=[color], s=12,
                  alpha=0.8 if label != -1 else 0.4)
    
    ax.set_title(f'Best Parameters: eps={best_eps:.3f}, min_pts={int(best_min_pts)}')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
