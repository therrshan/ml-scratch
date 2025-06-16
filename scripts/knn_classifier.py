import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

class KNNClassifier:
    """
    K-Nearest Neighbors Classifier implementation from scratch
    """
    
    def __init__(self, k=3, distance_metric='euclidean'):
        """
        Initialize KNN Classifier
        
        Parameters:
        k (int): Number of neighbors to consider
        distance_metric (str): Distance metric ('euclidean', 'manhattan', 'cosine')
        """
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None
        
    def _euclidean_distance(self, x1, x2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt(np.sum((x1 - x2)**2))
    
    def _manhattan_distance(self, x1, x2):
        """Calculate Manhattan distance between two points"""
        return np.sum(np.abs(x1 - x2))
    
    def _cosine_distance(self, x1, x2):
        """Calculate Cosine distance between two points"""
        dot_product = np.dot(x1, x2)
        norm_x1 = np.linalg.norm(x1)
        norm_x2 = np.linalg.norm(x2)
        
        if norm_x1 == 0 or norm_x2 == 0:
            return 1.0
        
        cosine_similarity = dot_product / (norm_x1 * norm_x2)
        return 1 - cosine_similarity
    
    def _calculate_distance(self, x1, x2):
        """Calculate distance based on chosen metric"""
        if self.distance_metric == 'euclidean':
            return self._euclidean_distance(x1, x2)
        elif self.distance_metric == 'manhattan':
            return self._manhattan_distance(x1, x2)
        elif self.distance_metric == 'cosine':
            return self._cosine_distance(x1, x2)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    def fit(self, X, y):
        """
        Train the KNN classifier (store training data)
        
        Parameters:
        X (array): Training features (m x n)
        y (array): Training labels (m,)
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        print(f"KNN model fitted with {len(self.X_train)} training samples")
    
    def _get_neighbors(self, x):
        """
        Get k nearest neighbors for a single point
        
        Parameters:
        x (array): Single data point
        
        Returns:
        list: Labels of k nearest neighbors
        """
        distances = []
        
        # Calculate distances to all training points
        for i, x_train in enumerate(self.X_train):
            dist = self._calculate_distance(x, x_train)
            distances.append((dist, self.y_train[i]))
        
        # Sort by distance and get k nearest
        distances.sort(key=lambda x: x[0])
        neighbors = [distances[i][1] for i in range(self.k)]
        
        return neighbors
    
    def predict_single(self, x):
        """
        Predict class for a single data point
        
        Parameters:
        x (array): Single data point
        
        Returns:
        Predicted class label
        """
        neighbors = self._get_neighbors(x)
        
        # Use majority voting
        prediction = Counter(neighbors).most_common(1)[0][0]
        return prediction
    
    def predict(self, X):
        """
        Predict classes for multiple data points
        
        Parameters:
        X (array): Features to predict on
        
        Returns:
        array: Predicted class labels
        """
        X = np.array(X)
        predictions = [self.predict_single(x) for x in X]
        return np.array(predictions)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for multiple data points
        
        Parameters:
        X (array): Features to predict on
        
        Returns:
        array: Class probabilities
        """
        X = np.array(X)
        probabilities = []
        
        # Get unique classes
        classes = np.unique(self.y_train)
        
        for x in X:
            neighbors = self._get_neighbors(x)
            neighbor_counts = Counter(neighbors)
            
            # Calculate probabilities for each class
            probs = []
            for cls in classes:
                probs.append(neighbor_counts.get(cls, 0) / self.k)
            
            probabilities.append(probs)
        
        return np.array(probabilities)
    
    def score(self, X, y):
        """
        Calculate accuracy score
        
        Parameters:
        X (array): Features
        y (array): True labels
        
        Returns:
        float: Accuracy score
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def plot_decision_boundary(self, X, y, resolution=0.02):
        """
        Plot decision boundary for 2D data
        
        Parameters:
        X (array): 2D feature array
        y (array): Labels
        resolution (float): Resolution of the decision boundary
        """
        if X.shape[1] != 2:
            print("Decision boundary plot only available for 2D data")
            return
        
        # Create color map
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        unique_labels = np.unique(y)
        color_map = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}
        
        plt.figure(figsize=(12, 8))
        
        # Plot training points
        for label in unique_labels:
            mask = y == label
            plt.scatter(X[mask, 0], X[mask, 1], 
                       c=color_map[label], label=f'Class {label}', alpha=0.7)
        
        # Create mesh for decision boundary
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                            np.arange(y_min, y_max, resolution))
        
        # Make predictions on mesh
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = self.predict(mesh_points)
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        plt.contourf(xx, yy, Z, alpha=0.3, levels=len(unique_labels)-1)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(f'KNN Decision Boundary (k={self.k}, metric={self.distance_metric})')
        plt.legend()
        plt.colorbar()
        plt.show()
    
    def cross_validate_k(self, X, y, k_range=range(1, 21), cv_folds=5):
        """
        Cross-validate to find optimal k value
        
        Parameters:
        X (array): Features
        y (array): Labels
        k_range (range): Range of k values to test
        cv_folds (int): Number of cross-validation folds
        
        Returns:
        dict: Results with k values and corresponding accuracies
        """
        results = {'k_values': [], 'accuracies': []}
        
        # Create CV folds
        n_samples = len(X)
        fold_size = n_samples // cv_folds
        
        for k in k_range:
            fold_accuracies = []
            
            for fold in range(cv_folds):
                # Create train/validation split
                val_start = fold * fold_size
                val_end = val_start + fold_size if fold < cv_folds - 1 else n_samples
                
                val_indices = list(range(val_start, val_end))
                train_indices = list(range(0, val_start)) + list(range(val_end, n_samples))
                
                X_train_fold = X[train_indices]
                y_train_fold = y[train_indices]
                X_val_fold = X[val_indices]
                y_val_fold = y[val_indices]
                
                # Train and evaluate
                knn_temp = KNNClassifier(k=k, distance_metric=self.distance_metric)
                knn_temp.fit(X_train_fold, y_train_fold)
                accuracy = knn_temp.score(X_val_fold, y_val_fold)
                fold_accuracies.append(accuracy)
            
            avg_accuracy = np.mean(fold_accuracies)
            results['k_values'].append(k)
            results['accuracies'].append(avg_accuracy)
            
            print(f"k={k}: CV Accuracy = {avg_accuracy:.4f}")
        
        # Plot results
        plt.figure(figsize=(10, 6))
        plt.plot(results['k_values'], results['accuracies'], 'bo-')
        plt.xlabel('k Value')
        plt.ylabel('Cross-Validation Accuracy')
        plt.title('KNN Cross-Validation Results')
        plt.grid(True)
        plt.show()
        
        # Find best k
        best_idx = np.argmax(results['accuracies'])
        best_k = results['k_values'][best_idx]
        best_accuracy = results['accuracies'][best_idx]
        
        print(f"\nBest k: {best_k} (Accuracy: {best_accuracy:.4f})")
        
        return results

# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=300, n_features=2, n_redundant=0,
                             n_informative=2, n_clusters_per_class=1, 
                             n_classes=3, random_state=42)
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Create and train model
    knn = KNNClassifier(k=5, distance_metric='euclidean')
    knn.fit(X_train, y_train)
    
    # Make predictions
    train_accuracy = knn.score(X_train, y_train)
    test_accuracy = knn.score(X_test, y_test)
    
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Plot decision boundary
    knn.plot_decision_boundary(X_train, y_train)
    
    # Cross-validate k value
    print("\nPerforming cross-validation to find optimal k...")
    results = knn.cross_validate_k(X_train, y_train, k_range=range(1, 16))