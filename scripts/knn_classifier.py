"""
K-Nearest Neighbors classifier implemented from scratch with multiple distance
metrics, cross-validation for k selection, and decision boundary visualization.
"""

import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

class KNNClassifier:
    
    def __init__(self, k=3, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None
        
    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))
    
    def _manhattan_distance(self, x1, x2):
        return np.sum(np.abs(x1 - x2))
    
    def _cosine_distance(self, x1, x2):
        dot_product = np.dot(x1, x2)
        norm_x1 = np.linalg.norm(x1)
        norm_x2 = np.linalg.norm(x2)
        if norm_x1 == 0 or norm_x2 == 0:
            return 1.0
        cosine_similarity = dot_product / (norm_x1 * norm_x2)
        return 1 - cosine_similarity
    
    def _calculate_distance(self, x1, x2):
        if self.distance_metric == 'euclidean':
            return self._euclidean_distance(x1, x2)
        elif self.distance_metric == 'manhattan':
            return self._manhattan_distance(x1, x2)
        elif self.distance_metric == 'cosine':
            return self._cosine_distance(x1, x2)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        print(f"KNN model fitted with {len(self.X_train)} training samples")
    
    def _get_neighbors(self, x):
        distances = []
        for i, x_train in enumerate(self.X_train):
            dist = self._calculate_distance(x, x_train)
            distances.append((dist, self.y_train[i]))
        distances.sort(key=lambda x: x[0])
        neighbors = [distances[i][1] for i in range(self.k)]
        return neighbors
    
    def predict_single(self, x):
        neighbors = self._get_neighbors(x)
        prediction = Counter(neighbors).most_common(1)[0][0]
        return prediction
    
    def predict(self, X):
        X = np.array(X)
        predictions = [self.predict_single(x) for x in X]
        return np.array(predictions)
    
    def predict_proba(self, X):
        X = np.array(X)
        probabilities = []
        classes = np.unique(self.y_train)
        for x in X:
            neighbors = self._get_neighbors(x)
            neighbor_counts = Counter(neighbors)
            probs = [neighbor_counts.get(cls, 0) / self.k for cls in classes]
            probabilities.append(probs)
        return np.array(probabilities)
    
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def plot_decision_boundary(self, X, y, resolution=0.02):
        if X.shape[1] != 2:
            print("Decision boundary plot only available for 2D data")
            return
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        unique_labels = np.unique(y)
        color_map = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}
        plt.figure(figsize=(12, 8))
        for label in unique_labels:
            mask = y == label
            plt.scatter(X[mask, 0], X[mask, 1], c=color_map[label], label=f'Class {label}', alpha=0.7)
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution), np.arange(y_min, y_max, resolution))
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = self.predict(mesh_points)
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.3, levels=len(unique_labels)-1)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(f'KNN Decision Boundary (k={self.k}, metric={self.distance_metric})')
        plt.legend()
        plt.colorbar()
        plt.show()
    
    def cross_validate_k(self, X, y, k_range=range(1, 21), cv_folds=5):
        results = {'k_values': [], 'accuracies': []}
        n_samples = len(X)
        fold_size = n_samples // cv_folds
        for k in k_range:
            fold_accuracies = []
            for fold in range(cv_folds):
                val_start = fold * fold_size
                val_end = val_start + fold_size if fold < cv_folds - 1 else n_samples
                val_indices = list(range(val_start, val_end))
                train_indices = list(range(0, val_start)) + list(range(val_end, n_samples))
                X_train_fold = X[train_indices]
                y_train_fold = y[train_indices]
                X_val_fold = X[val_indices]
                y_val_fold = y[val_indices]
                knn_temp = KNNClassifier(k=k, distance_metric=self.distance_metric)
                knn_temp.fit(X_train_fold, y_train_fold)
                accuracy = knn_temp.score(X_val_fold, y_val_fold)
                fold_accuracies.append(accuracy)
            avg_accuracy = np.mean(fold_accuracies)
            results['k_values'].append(k)
            results['accuracies'].append(avg_accuracy)
            print(f"k={k}: CV Accuracy = {avg_accuracy:.4f}")
        plt.figure(figsize=(10, 6))
        plt.plot(results['k_values'], results['accuracies'], 'bo-')
        plt.xlabel('k Value')
        plt.ylabel('Cross-Validation Accuracy')
        plt.title('KNN Cross-Validation Results')
        plt.grid(True)
        plt.show()
        best_idx = np.argmax(results['accuracies'])
        best_k = results['k_values'][best_idx]
        best_accuracy = results['accuracies'][best_idx]
        print(f"\nBest k: {best_k} (Accuracy: {best_accuracy:.4f})")
        return results

if __name__ == "__main__":
    np.random.seed(42)
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=300, n_features=2, n_redundant=0,
                             n_informative=2, n_clusters_per_class=1, 
                             n_classes=3, random_state=42)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    knn = KNNClassifier(k=5, distance_metric='euclidean')
    knn.fit(X_train, y_train)
    train_accuracy = knn.score(X_train, y_train)
    test_accuracy = knn.score(X_test, y_test)
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    knn.plot_decision_boundary(X_train, y_train)
    print("\nPerforming cross-validation to find optimal k...")
    results = knn.cross_validate_k(X_train, y_train, k_range=range(1, 16))
