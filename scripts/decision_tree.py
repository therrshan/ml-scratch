import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

class Node:
    """Node class for decision tree"""
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature      # Feature index to split on
        self.threshold = threshold  # Threshold value for split
        self.left = left           # Left child node
        self.right = right         # Right child node
        self.value = value         # Value if leaf node (class prediction)

class DecisionTreeClassifier:
    """
    Decision Tree Classifier implementation from scratch
    """
    
    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1, 
                 criterion='gini', max_features=None):
        """
        Initialize Decision Tree Classifier
        
        Parameters:
        max_depth (int): Maximum depth of the tree
        min_samples_split (int): Minimum samples required to split a node
        min_samples_leaf (int): Minimum samples required at a leaf node
        criterion (str): Splitting criterion ('gini' or 'entropy')
        max_features (int): Maximum features to consider for splitting
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.max_features = max_features
        self.root = None
        self.feature_importances_ = None
    
    def _gini_impurity(self, y):
        """Calculate Gini impurity"""
        if len(y) == 0:
            return 0
        
        proportions = np.array([np.sum(y == c) for c in np.unique(y)]) / len(y)
        return 1 - np.sum(proportions**2)
    
    def _entropy(self, y):
        """Calculate entropy"""
        if len(y) == 0:
            return 0
        
        proportions = np.array([np.sum(y == c) for c in np.unique(y)]) / len(y)
        proportions = proportions[proportions > 0]  # Remove zeros to avoid log(0)
        return -np.sum(proportions * np.log2(proportions))
    
    def _calculate_impurity(self, y):
        """Calculate impurity based on chosen criterion"""
        if self.criterion == 'gini':
            return self._gini_impurity(y)
        elif self.criterion == 'entropy':
            return self._entropy(y)
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")
    
    def _information_gain(self, X, y, feature, threshold):
        """Calculate information gain for a potential split"""
        # Parent impurity
        parent_impurity = self._calculate_impurity(y)
        
        # Split data
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return 0
        
        # Weighted average of children impurities
        n = len(y)
        left_impurity = self._calculate_impurity(y[left_mask])
        right_impurity = self._calculate_impurity(y[right_mask])
        
        weighted_impurity = (np.sum(left_mask) / n) * left_impurity + \
                           (np.sum(right_mask) / n) * right_impurity
        
        return parent_impurity - weighted_impurity
    
    def _best_split(self, X, y):
        """Find the best feature and threshold to split on"""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        # Determine features to consider
        if self.max_features is None:
            features_to_try = range(n_features)
        else:
            features_to_try = np.random.choice(n_features, 
                                             min(self.max_features, n_features), 
                                             replace=False)
        
        for feature in features_to_try:
            # Get unique values as potential thresholds
            unique_values = np.unique(X[:, feature])
            
            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2
                gain = self._information_gain(X, y, feature, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X, y, depth=0):
        """Recursively build the decision tree"""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Stopping criteria
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or
            n_classes == 1):
            # Return leaf node with most common class
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)
        
        # Find best split
        best_feature, best_threshold, best_gain = self._best_split(X, y)
        
        if best_gain == 0:
            # No good split found, create leaf
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)
        
        # Split the data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Check minimum samples per leaf
        if np.sum(left_mask) < self.min_samples_leaf or \
           np.sum(right_mask) < self.min_samples_leaf:
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)
        
        # Create child nodes
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return Node(feature=best_feature, threshold=best_threshold,
                   left=left_child, right=right_child)
    
    def fit(self, X, y):
        """
        Train the decision tree
        
        Parameters:
        X (array): Training features (m x n)
        y (array): Training labels (m,)
        """
        X = np.array(X)
        y = np.array(y)
        
        self.root = self._build_tree(X, y)
        self._calculate_feature_importances(X, y)
        
        print(f"Decision tree fitted with depth {self._get_tree_depth()}")
    
    def _predict_sample(self, x, node):
        """Predict class for a single sample"""
        if node.value is not None:
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)
    
    def predict(self, X):
        """
        Make predictions on new data
        
        Parameters:
        X (array): Features to predict on
        
        Returns:
        array: Predicted class labels
        """
        X = np.array(X)
        return np.array([self._predict_sample(x, self.root) for x in X])
    
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
    
    def _get_tree_depth(self, node=None):
        """Calculate the depth of the tree"""
        if node is None:
            node = self.root
        
        if node.value is not None:
            return 1
        
        left_depth = self._get_tree_depth(node.left)
        right_depth = self._get_tree_depth(node.right)
        
        return 1 + max(left_depth, right_depth)
    
    def _calculate_feature_importances(self, X, y):
        """Calculate feature importances"""
        n_features = X.shape[1]
        importances = np.zeros(n_features)
        
        def traverse(node, samples_at_node):
            if node.value is not None:
                return
            
            # Calculate importance for this node
            feature = node.feature
            n_samples = len(samples_at_node)
            
            if n_samples > 0:
                left_mask = X[samples_at_node, feature] <= node.threshold
                left_samples = samples_at_node[left_mask]
                right_samples = samples_at_node[~left_mask]
                
                if len(left_samples) > 0 and len(right_samples) > 0:
                    gain = self._information_gain(X[samples_at_node], 
                                                y[samples_at_node], 
                                                feature, node.threshold)
                    importances[feature] += gain * (n_samples / len(X))
                    
                    # Recursively traverse children
                    traverse(node.left, left_samples)
                    traverse(node.right, right_samples)
        
        traverse(self.root, np.arange(len(X)))
        
        # Normalize importances
        if np.sum(importances) > 0:
            importances = importances / np.sum(importances)
        
        self.feature_importances_ = importances
    
    def plot_feature_importances(self, feature_names=None):
        """Plot feature importances"""
        if self.feature_importances_ is None:
            print("Model must be fitted first")
            return
        
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(self.feature_importances_))]
        
        plt.figure(figsize=(10, 6))
        indices = np.argsort(self.feature_importances_)[::-1]
        
        plt.bar(range(len(self.feature_importances_)), 
                self.feature_importances_[indices])
        plt.xticks(range(len(self.feature_importances_)), 
                  [feature_names[i] for i in indices], rotation=45)
        plt.title('Feature Importances')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.show()
    
    def print_tree(self, node=None, depth=0):
        """Print the tree structure"""
        if node is None:
            node = self.root
        
        if node.value is not None:
            print('  ' * depth + f'Predict: {node.value}')
        else:
            print('  ' * depth + f'Feature {node.feature} <= {node.threshold:.3f}')
            print('  ' * depth + 'Left:')
            self.print_tree(node.left, depth + 1)
            print('  ' * depth + 'Right:')
            self.print_tree(node.right, depth + 1)
    
    def plot_decision_boundary(self, X, y, resolution=0.02):
        """Plot decision boundary for 2D data"""
        if X.shape[1] != 2:
            print("Decision boundary plot only available for 2D data")
            return
        
        # Create color map
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        unique_labels = np.unique(y)
        color_map = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}
        
        plt.figure(figsize=(12, 8))
        
        # Plot data points
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
        plt.title(f'Decision Tree Decision Boundary (depth={self._get_tree_depth()})')
        plt.legend()
        plt.colorbar()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=500, n_features=4, n_redundant=0,
                             n_informative=4, n_clusters_per_class=1, 
                             n_classes=3, random_state=42)
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Create and train model
    dt = DecisionTreeClassifier(max_depth=5, min_samples_split=10, 
                               min_samples_leaf=5, criterion='gini')
    dt.fit(X_train, y_train)
    
    # Make predictions
    train_accuracy = dt.score(X_train, y_train)
    test_accuracy = dt.score(X_test, y_test)
    
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Tree Depth: {dt._get_tree_depth()}")
    
    # Print tree structure (first few levels)
    print("\nTree Structure:")
    dt.print_tree()
    
    # Plot feature importances
    dt.plot_feature_importances()
    
    # For 2D visualization, create a simpler dataset
    X_2d, y_2d = make_classification(n_samples=300, n_features=2, n_redundant=0,
                                   n_informative=2, n_clusters_per_class=1, 
                                   n_classes=3, random_state=42)
    
    dt_2d = DecisionTreeClassifier(max_depth=5, criterion='gini')
    dt_2d.fit(X_2d, y_2d)
    dt_2d.plot_decision_boundary(X_2d, y_2d)