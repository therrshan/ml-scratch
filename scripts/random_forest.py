"""
Random Forest classifier implemented from scratch with bootstrap sampling,
feature importance analysis, and visualization capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import random

class DecisionTreeRF:
    
    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1, 
                 max_features=None, criterion='gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.criterion = criterion
        self.tree = None
        
    def _gini_impurity(self, y):
        if len(y) == 0:
            return 0
        proportions = np.array([np.sum(y == c) for c in np.unique(y)]) / len(y)
        return 1 - np.sum(proportions**2)
    
    def _entropy(self, y):
        if len(y) == 0:
            return 0
        proportions = np.array([np.sum(y == c) for c in np.unique(y)]) / len(y)
        proportions = proportions[proportions > 0]
        return -np.sum(proportions * np.log2(proportions))
    
    def _calculate_impurity(self, y):
        if self.criterion == 'gini':
            return self._gini_impurity(y)
        else:
            return self._entropy(y)
    
    def _best_split(self, X, y, feature_indices):
        best_gain = 0
        best_feature = None
        best_threshold = None
        
        for feature in feature_indices:
            unique_values = np.unique(X[:, feature])
            
            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2
                
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                parent_impurity = self._calculate_impurity(y)
                n = len(y)
                left_impurity = self._calculate_impurity(y[left_mask])
                right_impurity = self._calculate_impurity(y[right_mask])
                
                weighted_impurity = (np.sum(left_mask) / n) * left_impurity + \
                                   (np.sum(right_mask) / n) * right_impurity
                
                gain = parent_impurity - weighted_impurity
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or
            n_classes == 1):
            leaf_value = Counter(y).most_common(1)[0][0]
            return {'leaf': True, 'value': leaf_value}
        
        if self.max_features is None:
            feature_indices = list(range(n_features))
        else:
            feature_indices = random.sample(range(n_features), 
                                          min(self.max_features, n_features))
        
        best_feature, best_threshold, best_gain = self._best_split(X, y, feature_indices)
        
        if best_gain == 0:
            leaf_value = Counter(y).most_common(1)[0][0]
            return {'leaf': True, 'value': leaf_value}
        
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        if np.sum(left_mask) < self.min_samples_leaf or \
           np.sum(right_mask) < self.min_samples_leaf:
            leaf_value = Counter(y).most_common(1)[0][0]
            return {'leaf': True, 'value': leaf_value}
        
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {
            'leaf': False,
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_subtree,
            'right': right_subtree
        }
    
    def fit(self, X, y):
        self.tree = self._build_tree(X, y)
    
    def _predict_sample(self, x, tree):
        if tree['leaf']:
            return tree['value']
        
        if x[tree['feature']] <= tree['threshold']:
            return self._predict_sample(x, tree['left'])
        else:
            return self._predict_sample(x, tree['right'])
    
    def predict(self, X):
        return np.array([self._predict_sample(x, self.tree) for x in X])


class RandomForest:
    
    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=2,
                 min_samples_leaf=1, max_features='sqrt', bootstrap=True,
                 criterion='gini', random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.criterion = criterion
        self.random_state = random_state
        
        self.trees = []
        self.feature_importances_ = None
        
        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)
    
    def _get_max_features(self, n_features):
        if self.max_features == 'sqrt':
            return int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            return int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        else:
            return n_features
    
    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[bootstrap_indices], y[bootstrap_indices], bootstrap_indices
    
    def _calculate_oob_score(self, X, y, tree_predictions, bootstrap_indices_list):
        oob_predictions = np.full(len(y), -1)
        oob_counts = np.zeros(len(y))
        
        for i, (tree, bootstrap_indices) in enumerate(zip(self.trees, bootstrap_indices_list)):
            oob_indices = np.setdiff1d(np.arange(len(y)), bootstrap_indices)
            
            if len(oob_indices) > 0:
                oob_pred = tree.predict(X[oob_indices])
                
                for j, idx in enumerate(oob_indices):
                    oob_predictions[idx] = oob_pred[j]
                    oob_counts[idx] += 1
        
        valid_oob = oob_counts > 0
        if np.sum(valid_oob) > 0:
            oob_accuracy = np.mean(oob_predictions[valid_oob] == y[valid_oob])
            return oob_accuracy
        else:
            return 0.0
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape
        
        max_features = self._get_max_features(n_features)
        self.trees = []
        bootstrap_indices_list = []
        
        print(f"Training Random Forest with {self.n_estimators} trees...")
        print(f"Using {max_features} features per tree out of {n_features} total features")
        
        for i in range(self.n_estimators):
            if (i + 1) % 20 == 0:
                print(f"Training tree {i + 1}/{self.n_estimators}")
            
            if self.bootstrap:
                X_bootstrap, y_bootstrap, bootstrap_indices = self._bootstrap_sample(X, y)
                bootstrap_indices_list.append(bootstrap_indices)
            else:
                X_bootstrap, y_bootstrap = X, y
                bootstrap_indices_list.append(np.arange(len(y)))
            
            tree = DecisionTreeRF(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_features,
                criterion=self.criterion
            )
            
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)
        
        self._calculate_feature_importances(X, y)
        
        if self.bootstrap:
            oob_score = self._calculate_oob_score(X, y, None, bootstrap_indices_list)
            print(f"Out-of-Bag Score: {oob_score:.4f}")
        
        print("Random Forest training completed!")
    
    def _calculate_feature_importances(self, X, y):
        n_features = X.shape[1]
        importances = np.zeros(n_features)
        
        for feature_idx in range(n_features):
            feature_importance = 0
            
            for tree in self.trees:
                feature_importance += self._count_feature_usage(tree.tree, feature_idx)
            
            importances[feature_idx] = feature_importance
        
        if np.sum(importances) > 0:
            importances = importances / np.sum(importances)
        
        self.feature_importances_ = importances
    
    def _count_feature_usage(self, tree_node, feature_idx):
        if tree_node['leaf']:
            return 0
        
        count = 1 if tree_node['feature'] == feature_idx else 0
        count += self._count_feature_usage(tree_node['left'], feature_idx)
        count += self._count_feature_usage(tree_node['right'], feature_idx)
        
        return count
    
    def predict(self, X):
        X = np.array(X)
        
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        
        predictions = []
        for i in range(X.shape[0]):
            votes = tree_predictions[:, i]
            prediction = Counter(votes).most_common(1)[0][0]
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        X = np.array(X)
        
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        
        unique_classes = np.unique(tree_predictions)
        n_classes = len(unique_classes)
        
        probabilities = np.zeros((X.shape[0], n_classes))
        
        for i in range(X.shape[0]):
            votes = tree_predictions[:, i]
            vote_counts = Counter(votes)
            
            for j, cls in enumerate(unique_classes):
                probabilities[i, j] = vote_counts.get(cls, 0) / len(self.trees)
        
        return probabilities
    
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def plot_feature_importances(self, feature_names=None, top_n=None):
        if self.feature_importances_ is None:
            print("Model must be fitted first")
            return
        
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(self.feature_importances_))]
        
        indices = np.argsort(self.feature_importances_)[::-1]
        
        if top_n is not None:
            indices = indices[:top_n]
        
        plt.figure(figsize=(12, 8))
        plt.bar(range(len(indices)), self.feature_importances_[indices])
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45)
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title('Random Forest Feature Importances')
        plt.tight_layout()
        plt.show()
    
    def plot_decision_boundary(self, X, y, resolution=0.02):
        if X.shape[1] != 2:
            print("Decision boundary plot only available for 2D data")
            return
        
        plt.figure(figsize=(12, 8))
        
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        unique_labels = np.unique(y)
        
        for i, label in enumerate(unique_labels):
            mask = y == label
            plt.scatter(X[mask, 0], X[mask, 1], 
                       c=colors[i % len(colors)], alpha=0.7, s=50, 
                       label=f'Class {label}')
        
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                            np.arange(y_min, y_max, resolution))
        
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = self.predict(mesh_points)
        Z = Z.reshape(xx.shape)
        
        plt.contourf(xx, yy, Z, alpha=0.3, levels=len(unique_labels)-1)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(f'Random Forest Decision Boundary ({self.n_estimators} trees)')
        plt.legend()
        plt.colorbar()
        plt.show()
    
    def plot_tree_predictions(self, X, y, n_trees_to_show=4):
        if X.shape[1] != 2:
            print("Tree prediction plot only available for 2D data")
            return
        
        n_trees_to_show = min(n_trees_to_show, len(self.trees))
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        unique_labels = np.unique(y)
        
        for i in range(n_trees_to_show):
            ax = axes[i]
            
            tree_pred = self.trees[i].predict(X)
            
            for j, label in enumerate(unique_labels):
                mask = tree_pred == label
                ax.scatter(X[mask, 0], X[mask, 1], 
                          c=colors[j % len(colors)], alpha=0.7, s=30)
            
            ax.set_title(f'Tree {i+1} Predictions')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.grid(True, alpha=0.3)
        
        ax = axes[4]
        ensemble_pred = self.predict(X)
        
        for j, label in enumerate(unique_labels):
            mask = ensemble_pred == label
            ax.scatter(X[mask, 0], X[mask, 1], 
                      c=colors[j % len(colors)], alpha=0.7, s=30)
        
        ax.set_title('Ensemble Predictions')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.grid(True, alpha=0.3)
        
        ax = axes[5]
        for j, label in enumerate(unique_labels):
            mask = y == label
            ax.scatter(X[mask, 0], X[mask, 1], 
                      c=colors[j % len(colors)], alpha=0.7, s=30, 
                      label=f'Class {label}')
        
        ax.set_title('True Labels')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    print("=== Random Forest Classifier Example ===")
    
    np.random.seed(42)
    from sklearn.datasets import make_classification, make_circles, make_moons
    
    X_class, y_class = make_classification(n_samples=1000, n_features=10, 
                                         n_informative=8, n_redundant=2,
                                         n_classes=3, n_clusters_per_class=1, 
                                         random_state=42)
    
    split_idx = int(0.8 * len(X_class))
    X_train, X_test = X_class[:split_idx], X_class[split_idx:]
    y_train, y_test = y_class[:split_idx], y_class[split_idx:]
    
    rf = RandomForest(
        n_estimators=50,
        max_depth=8,
        max_features='sqrt',
        bootstrap=True,
        criterion='gini',
        random_state=42
    )
    
    rf.fit(X_train, y_train)
    
    train_accuracy = rf.score(X_train, y_train)
    test_accuracy = rf.score(X_test, y_test)
    
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    feature_names = [f'Feature_{i}' for i in range(X_class.shape[1])]
    rf.plot_feature_importances(feature_names, top_n=8)
    
    print("\n=== 2D Visualization Example ===")
    X_2d, y_2d = make_circles(n_samples=400, noise=0.1, factor=0.3, random_state=42)
    
    rf_2d = RandomForest(
        n_estimators=30,
        max_depth=6,
        max_features='sqrt',
        random_state=42
    )
    
    rf_2d.fit(X_2d, y_2d)
    
    print(f"2D Dataset Accuracy: {rf_2d.score(X_2d, y_2d):.4f}")
    
    rf_2d.plot_decision_boundary(X_2d, y_2d)
    
    rf_2d.plot_tree_predictions(X_2d, y_2d, n_trees_to_show=4)
    
    print("\n=== Comparing Different Configurations ===")
    
    n_estimators_list = [5, 10, 25, 50, 100]
    estimator_results = []
    
    for n_est in n_estimators_list:
        rf_comp = RandomForest(n_estimators=n_est, max_depth=6, random_state=42)
        rf_comp.fit(X_train, y_train)
        accuracy = rf_comp.score(X_test, y_test)
        estimator_results.append((n_est, accuracy))
        print(f"n_estimators={n_est}: Test Accuracy = {accuracy:.4f}")
    
    n_est_values = [x[0] for x in estimator_results]
    accuracies = [x[1] for x in estimator_results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_est_values, accuracies, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Estimators')
    plt.ylabel('Test Accuracy')
    plt.title('Random Forest: Effect of Number of Trees')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("\n=== Max Features Comparison ===")
    
    max_features_options = ['sqrt', 'log2', None, 5]
    feature_results = []
    
    for max_feat in max_features_options:
        rf_feat = RandomForest(n_estimators=30, max_depth=6, 
                              max_features=max_feat, random_state=42)
        rf_feat.fit(X_train, y_train)
        accuracy = rf_feat.score(X_test, y_test)
        feature_results.append((str(max_feat), accuracy))
        print(f"max_features={max_feat}: Test Accuracy = {accuracy:.4f}")
    
    feat_names = [x[0] for x in feature_results]
    feat_accuracies = [x[1] for x in feature_results]
    
    plt.figure(figsize=(10, 6))
    plt.bar(feat_names, feat_accuracies, alpha=0.7, color=['skyblue', 'lightgreen', 'orange', 'pink'])
    plt.xlabel('Max Features')
    plt.ylabel('Test Accuracy')
    plt.title('Random Forest: Effect of Max Features')
    plt.grid(True, alpha=0.3, axis='y')
    
    for i, acc in enumerate(feat_accuracies):
        plt.text(i, acc + 0.005, f'{acc:.3f}', ha='center', va='bottom')
    
    plt.show()
    
    print("\n=== Bootstrap vs No Bootstrap ===")
    
    rf_bootstrap = RandomForest(n_estimators=30, bootstrap=True, random_state=42)
    rf_no_bootstrap = RandomForest(n_estimators=30, bootstrap=False, random_state=42)
    
    rf_bootstrap.fit(X_train, y_train)
    rf_no_bootstrap.fit(X_train, y_train)
    
    bootstrap_acc = rf_bootstrap.score(X_test, y_test)
    no_bootstrap_acc = rf_no_bootstrap.score(X_test, y_test)
    
    print(f"With Bootstrap: {bootstrap_acc:.4f}")
    print(f"Without Bootstrap: {no_bootstrap_acc:.4f}")
    
    print("\n=== Moons Dataset Example ===")
    
    X_moons, y_moons = make_moons(n_samples=300, noise=0.2, random_state=42)
    
    rf_moons = RandomForest(n_estimators=50, max_depth=8, random_state=42)
    rf_moons.fit(X_moons, y_moons)
    
    moons_accuracy = rf_moons.score(X_moons, y_moons)
    print(f"Moons Dataset Accuracy: {moons_accuracy:.4f}")
    
    rf_moons.plot_decision_boundary(X_moons, y_moons)