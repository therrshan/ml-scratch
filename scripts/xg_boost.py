"""
XGBoost implementation from scratch with gradient boosting, regularization,
subsampling, early stopping, and feature importance analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class XGBoostTree:
    
    def __init__(self, max_depth=6, min_child_weight=1, reg_lambda=1, reg_alpha=0, gamma=0):
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.gamma = gamma
        self.tree = None
        
    def _calculate_leaf_weight(self, gradients, hessians):
        G = np.sum(gradients)
        H = np.sum(hessians)
        
        if self.reg_alpha > 0:
            if G > self.reg_alpha:
                numerator = G - self.reg_alpha
            elif G < -self.reg_alpha:
                numerator = G + self.reg_alpha
            else:
                numerator = 0
        else:
            numerator = G
        
        denominator = H + self.reg_lambda
        
        if denominator == 0:
            return 0
        
        return -numerator / denominator
    
    def _calculate_gain(self, left_gradients, left_hessians, right_gradients, right_hessians):
        def calculate_score(gradients, hessians):
            G = np.sum(gradients)
            H = np.sum(hessians)
            
            if self.reg_alpha > 0:
                if G > self.reg_alpha:
                    numerator = (G - self.reg_alpha) ** 2
                elif G < -self.reg_alpha:
                    numerator = (G + self.reg_alpha) ** 2
                else:
                    numerator = 0
            else:
                numerator = G ** 2
            
            denominator = H + self.reg_lambda
            return numerator / denominator if denominator > 0 else 0
        
        left_score = calculate_score(left_gradients, left_hessians)
        right_score = calculate_score(right_gradients, right_hessians)
        parent_score = calculate_score(
            np.concatenate([left_gradients, right_gradients]),
            np.concatenate([left_hessians, right_hessians])
        )
        
        gain = 0.5 * (left_score + right_score - parent_score) - self.gamma
        return gain
    
    def _find_best_split(self, X, gradients, hessians):
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        n_samples, n_features = X.shape
        
        for feature in range(n_features):
            unique_values = np.unique(X[:, feature])
            
            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2
                
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                left_hessian_sum = np.sum(hessians[left_mask])
                right_hessian_sum = np.sum(hessians[right_mask])
                
                if (left_hessian_sum < self.min_child_weight or 
                    right_hessian_sum < self.min_child_weight):
                    continue
                
                gain = self._calculate_gain(
                    gradients[left_mask], hessians[left_mask],
                    gradients[right_mask], hessians[right_mask]
                )
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X, gradients, hessians, depth=0):
        if (depth >= self.max_depth or 
            len(gradients) == 0 or
            np.sum(hessians) < self.min_child_weight):
            
            leaf_weight = self._calculate_leaf_weight(gradients, hessians)
            return {'leaf': True, 'weight': leaf_weight}
        
        best_feature, best_threshold, best_gain = self._find_best_split(X, gradients, hessians)
        
        if best_gain <= 0:
            leaf_weight = self._calculate_leaf_weight(gradients, hessians)
            return {'leaf': True, 'weight': leaf_weight}
        
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        left_subtree = self._build_tree(
            X[left_mask], gradients[left_mask], hessians[left_mask], depth + 1
        )
        right_subtree = self._build_tree(
            X[right_mask], gradients[right_mask], hessians[right_mask], depth + 1
        )
        
        return {
            'leaf': False,
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_subtree,
            'right': right_subtree,
            'gain': best_gain
        }
    
    def fit(self, X, gradients, hessians):
        self.tree = self._build_tree(X, gradients, hessians)
    
    def _predict_sample(self, x, tree):
        if tree['leaf']:
            return tree['weight']
        
        if x[tree['feature']] <= tree['threshold']:
            return self._predict_sample(x, tree['left'])
        else:
            return self._predict_sample(x, tree['right'])
    
    def predict(self, X):
        return np.array([self._predict_sample(x, self.tree) for x in X])


class XGBoost:
    
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.3,
                 min_child_weight=1, subsample=1.0, colsample_bytree=1.0,
                 reg_lambda=1, reg_alpha=0, gamma=0, objective='binary:logistic',
                 eval_metric='error', early_stopping_rounds=None, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.gamma = gamma
        self.objective = objective
        self.eval_metric = eval_metric
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        
        self.trees = []
        self.base_prediction = 0
        self.train_scores = []
        self.val_scores = []
        self.feature_importances_ = None
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _sigmoid(self, x):
        x = np.clip(x, -250, 250)
        return 1 / (1 + np.exp(-x))
    
    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def _calculate_gradients_hessians(self, y_true, y_pred):
        if self.objective == 'binary:logistic':
            pred_prob = self._sigmoid(y_pred)
            gradients = pred_prob - y_true
            hessians = pred_prob * (1 - pred_prob)
            
        elif self.objective == 'reg:squarederror':
            gradients = y_pred - y_true
            hessians = np.ones_like(y_true)
            
        elif self.objective == 'multi:softmax':
            pred_prob = self._softmax(y_pred.reshape(-1, 1))
            gradients = pred_prob.flatten() - y_true
            hessians = pred_prob.flatten() * (1 - pred_prob.flatten())
            
        else:
            raise ValueError(f"Unknown objective: {self.objective}")
        
        return gradients, hessians
    
    def _calculate_metric(self, y_true, y_pred):
        if self.eval_metric == 'error':
            if self.objective == 'binary:logistic':
                pred_class = (self._sigmoid(y_pred) > 0.5).astype(int)
                return 1 - np.mean(pred_class == y_true)
            elif self.objective == 'multi:softmax':
                pred_class = np.argmax(y_pred, axis=1)
                return 1 - np.mean(pred_class == y_true)
            else:
                return np.sqrt(np.mean((y_true - y_pred)**2))
        
        elif self.eval_metric == 'logloss':
            if self.objective == 'binary:logistic':
                pred_prob = self._sigmoid(y_pred)
                pred_prob = np.clip(pred_prob, 1e-15, 1 - 1e-15)
                return -np.mean(y_true * np.log(pred_prob) + (1 - y_true) * np.log(1 - pred_prob))
        
        elif self.eval_metric == 'rmse':
            return np.sqrt(np.mean((y_true - y_pred)**2))
        
        return 0
    
    def _subsample_data(self, X, y, gradients, hessians):
        n_samples = X.shape[0]
        n_features = X.shape[1]
        
        if self.subsample < 1.0:
            n_subsample = int(n_samples * self.subsample)
            indices = np.random.choice(n_samples, n_subsample, replace=False)
            X_sub = X[indices]
            y_sub = y[indices]
            grad_sub = gradients[indices]
            hess_sub = hessians[indices]
        else:
            X_sub = X
            y_sub = y
            grad_sub = gradients
            hess_sub = hessians
        
        if self.colsample_bytree < 1.0:
            n_features_sub = int(n_features * self.colsample_bytree)
            feature_indices = np.random.choice(n_features, n_features_sub, replace=False)
            X_sub = X_sub[:, feature_indices]
        else:
            feature_indices = np.arange(n_features)
        
        return X_sub, y_sub, grad_sub, hess_sub, feature_indices
    
    def fit(self, X, y, eval_set=None, verbose=True):
        X = np.array(X)
        y = np.array(y)
        
        if self.objective == 'binary:logistic':
            self.base_prediction = np.log(np.mean(y) / (1 - np.mean(y) + 1e-15))
        elif self.objective == 'reg:squarederror':
            self.base_prediction = np.mean(y)
        else:
            self.base_prediction = 0
        
        train_pred = np.full(len(y), self.base_prediction)
        
        if eval_set is not None:
            X_val, y_val = eval_set
            val_pred = np.full(len(y_val), self.base_prediction)
        
        self.trees = []
        best_score = float('inf')
        rounds_without_improvement = 0
        
        if verbose:
            print(f"Training XGBoost with {self.n_estimators} rounds...")
        
        for round_num in range(self.n_estimators):
            gradients, hessians = self._calculate_gradients_hessians(y, train_pred)
            
            X_sub, y_sub, grad_sub, hess_sub, feature_indices = self._subsample_data(
                X, y, gradients, hessians
            )
            
            tree = XGBoostTree(
                max_depth=self.max_depth,
                min_child_weight=self.min_child_weight,
                reg_lambda=self.reg_lambda,
                reg_alpha=self.reg_alpha,
                gamma=self.gamma
            )
            
            tree.fit(X_sub, grad_sub, hess_sub)
            
            self.trees.append((tree, feature_indices))
            
            if self.colsample_bytree < 1.0:
                tree_pred = tree.predict(X[:, feature_indices])
            else:
                tree_pred = tree.predict(X)
            
            train_pred += self.learning_rate * tree_pred
            
            train_score = self._calculate_metric(y, train_pred)
            self.train_scores.append(train_score)
            
            if eval_set is not None:
                if self.colsample_bytree < 1.0:
                    val_tree_pred = tree.predict(X_val[:, feature_indices])
                else:
                    val_tree_pred = tree.predict(X_val)
                    
                val_pred += self.learning_rate * val_tree_pred
                val_score = self._calculate_metric(y_val, val_pred)
                self.val_scores.append(val_score)
                
                if self.early_stopping_rounds is not None:
                    if val_score < best_score:
                        best_score = val_score
                        rounds_without_improvement = 0
                    else:
                        rounds_without_improvement += 1
                        
                    if rounds_without_improvement >= self.early_stopping_rounds:
                        if verbose:
                            print(f"Early stopping at round {round_num}")
                        break
            
            if verbose and (round_num + 1) % 10 == 0:
                if eval_set is not None:
                    print(f"Round {round_num + 1}: train-{self.eval_metric}={train_score:.6f}, "
                          f"val-{self.eval_metric}={val_score:.6f}")
                else:
                    print(f"Round {round_num + 1}: train-{self.eval_metric}={train_score:.6f}")
        
        self._calculate_feature_importances(X.shape[1])
        
        if verbose:
            print("XGBoost training completed!")
    
    def _calculate_feature_importances(self, n_features):
        importances = np.zeros(n_features)
        
        for tree, feature_indices in self.trees:
            tree_importances = self._get_tree_importances(tree.tree, feature_indices, n_features)
            importances += tree_importances
        
        if np.sum(importances) > 0:
            importances = importances / np.sum(importances)
        
        self.feature_importances_ = importances
    
    def _get_tree_importances(self, tree_node, feature_indices, n_features):
        importances = np.zeros(n_features)
        
        if tree_node['leaf']:
            return importances
        
        original_feature_idx = feature_indices[tree_node['feature']]
        importances[original_feature_idx] += tree_node['gain']
        
        importances += self._get_tree_importances(tree_node['left'], feature_indices, n_features)
        importances += self._get_tree_importances(tree_node['right'], feature_indices, n_features)
        
        return importances
    
    def predict(self, X):
        X = np.array(X)
        predictions = np.full(X.shape[0], self.base_prediction)
        
        for tree, feature_indices in self.trees:
            if self.colsample_bytree < 1.0:
                tree_pred = tree.predict(X[:, feature_indices])
            else:
                tree_pred = tree.predict(X)
            predictions += self.learning_rate * tree_pred
        
        if self.objective == 'binary:logistic':
            return (self._sigmoid(predictions) > 0.5).astype(int)
        elif self.objective == 'reg:squarederror':
            return predictions
        else:
            return predictions
    
    def predict_proba(self, X):
        X = np.array(X)
        predictions = np.full(X.shape[0], self.base_prediction)
        
        for tree, feature_indices in self.trees:
            if self.colsample_bytree < 1.0:
                tree_pred = tree.predict(X[:, feature_indices])
            else:
                tree_pred = tree.predict(X)
            predictions += self.learning_rate * tree_pred
        
        if self.objective == 'binary:logistic':
            prob_pos = self._sigmoid(predictions)
            return np.column_stack([1 - prob_pos, prob_pos])
        else:
            return predictions
    
    def score(self, X, y):
        if self.objective == 'binary:logistic':
            predictions = self.predict(X)
            return np.mean(predictions == y)
        else:
            predictions = self.predict(X)
            return -np.mean((y - predictions)**2)
    
    def plot_training_history(self):
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_scores, label='Training')
        if self.val_scores:
            plt.plot(self.val_scores, label='Validation')
        plt.xlabel('Boosting Round')
        plt.ylabel(f'{self.eval_metric}')
        plt.title('Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        if self.feature_importances_ is not None:
            top_features = np.argsort(self.feature_importances_)[-10:]
            plt.barh(range(len(top_features)), self.feature_importances_[top_features])
            plt.yticks(range(len(top_features)), [f'Feature {i}' for i in top_features])
            plt.xlabel('Importance')
            plt.title('Top 10 Feature Importances')
        
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importances(self, feature_names=None, top_n=20):
        if self.feature_importances_ is None:
            print("Model must be fitted first")
            return
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(self.feature_importances_))]
        
        top_indices = np.argsort(self.feature_importances_)[-top_n:]
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_indices)), self.feature_importances_[top_indices])
        plt.yticks(range(len(top_indices)), [feature_names[i] for i in top_indices])
        plt.xlabel('Importance')
        plt.title('XGBoost Feature Importances')
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
        plt.title(f'XGBoost Decision Boundary ({len(self.trees)} trees)')
        plt.legend()
        plt.colorbar()
        plt.show()


if __name__ == "__main__":
    print("=== XGBoost Classifier Example ===")
    
    np.random.seed(42)
    from sklearn.datasets import make_classification, make_regression, make_circles
    
    X_class, y_class = make_classification(n_samples=1000, n_features=10, 
                                         n_informative=8, n_redundant=2,
                                         n_classes=2, n_clusters_per_class=1, 
                                         random_state=42)
    
    split_idx = int(0.8 * len(X_class))
    X_train, X_test = X_class[:split_idx], X_class[split_idx:]
    y_train, y_test = y_class[:split_idx], y_class[split_idx:]
    
    val_split = int(0.8 * len(X_train))
    X_train_sub, X_val = X_train[:val_split], X_train[val_split:]
    y_train_sub, y_val = y_train[:val_split], y_train[val_split:]
    
    xgb = XGBoost(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1,
        reg_alpha=0,
        objective='binary:logistic',
        eval_metric='error',
        early_stopping_rounds=10,
        random_state=42
    )
    
    xgb.fit(X_train_sub, y_train_sub, eval_set=(X_val, y_val))
    
    train_accuracy = xgb.score(X_train, y_train)
    test_accuracy = xgb.score(X_test, y_test)
    
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    xgb.plot_training_history()
    
    feature_names = [f'Feature_{i}' for i in range(X_class.shape[1])]
    xgb.plot_feature_importances(feature_names)
    
    print("\n=== XGBoost Regression Example ===")
    
    X_reg, y_reg = make_regression(n_samples=500, n_features=8, noise=0.1, random_state=42)
    
    split_idx = int(0.8 * len(X_reg))
    X_train_reg, X_test_reg = X_reg[:split_idx], X_reg[split_idx:]
    y_train_reg, y_test_reg = y_reg[:split_idx], y_reg[split_idx:]
    
    xgb_reg = XGBoost(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        objective='reg:squarederror',
        eval_metric='rmse',
        random_state=42
    )
    
    xgb_reg.fit(X_train_reg, y_train_reg)
    
    train_pred = xgb_reg.predict(X_train_reg)
    test_pred = xgb_reg.predict(X_test_reg)
    
    train_rmse = np.sqrt(np.mean((y_train_reg - train_pred)**2))
    test_rmse = np.sqrt(np.mean((y_test_reg - test_pred)**2))
    
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test_reg, test_pred, alpha=0.6)
    plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('XGBoost Regression: Predictions vs True Values')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(xgb_reg.train_scores)
    plt.xlabel('Boosting Round')
    plt.ylabel('RMSE')
    plt.title('Training RMSE')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== Subsample and Colsample Comparison ===")
    
    sampling_settings = [
        ('Full', 1.0, 1.0),
        ('Sub 0.8', 0.8, 1.0),
        ('ColSample 0.8', 1.0, 0.8),
        ('Both 0.8', 0.8, 0.8),
        ('Both 0.6', 0.6, 0.6)
    ]
    
    sampling_results = []
    
    for name, subsample, colsample in sampling_settings:
        xgb_sample = XGBoost(
            n_estimators=50,
            max_depth=6,
            learning_rate=0.1,
            subsample=subsample,
            colsample_bytree=colsample,
            random_state=42
        )
        xgb_sample.fit(X_train_sub, y_train_sub, verbose=False)
        train_acc = xgb_sample.score(X_train_sub, y_train_sub)
        test_acc = xgb_sample.score(X_test, y_test)
        sampling_results.append((name, train_acc, test_acc))
        print(f"{name}: Train={train_acc:.4f}, Test={test_acc:.4f}")
    
    sample_names = [x[0] for x in sampling_results]
    sample_train_accs = [x[1] for x in sampling_results]
    sample_test_accs = [x[2] for x in sampling_results]
    
    x = np.arange(len(sample_names))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, sample_train_accs, width, label='Train', alpha=0.7)
    plt.bar(x + width/2, sample_test_accs, width, label='Test', alpha=0.7)
    plt.xlabel('Sampling Setting')
    plt.ylabel('Accuracy')
    plt.title('XGBoost: Effect of Subsampling')
    plt.xticks(x, sample_names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()
    
    print("\n=== Early Stopping Demonstration ===")
    
    xgb_early = XGBoost(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        objective='binary:logistic',
        eval_metric='error',
        early_stopping_rounds=15,
        random_state=42
    )
    
    print("Training with early stopping...")
    xgb_early.fit(X_train_sub, y_train_sub, eval_set=(X_val, y_val))
    
    print(f"Stopped at {len(xgb_early.trees)} trees out of {xgb_early.n_estimators}")
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(xgb_early.train_scores, label='Training Error')
    plt.plot(xgb_early.val_scores, label='Validation Error')
    plt.xlabel('Boosting Round')
    plt.ylabel('Error Rate')
    plt.title('Early Stopping Demonstration')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    xgb_no_early = XGBoost(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        objective='binary:logistic',
        eval_metric='error',
        early_stopping_rounds=None,
        random_state=42
    )
    
    xgb_no_early.fit(X_train_sub, y_train_sub, eval_set=(X_val, y_val), verbose=False)
    
    plt.subplot(1, 2, 2)
    plt.plot(xgb_no_early.train_scores, label='Training Error')
    plt.plot(xgb_no_early.val_scores, label='Validation Error')
    plt.xlabel('Boosting Round')
    plt.ylabel('Error Rate')
    plt.title('Without Early Stopping')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Early Stopping Test Accuracy: {xgb_early.score(X_test, y_test):.4f}")
    print(f"No Early Stopping Test Accuracy: {xgb_no_early.score(X_test, y_test):.4f}")
    
    print("\n=== Probability Calibration Analysis ===")
    
    xgb_prob = XGBoost(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        objective='binary:logistic',
        random_state=42
    )
    
    xgb_prob.fit(X_train, y_train, verbose=False)
    
    train_probs = xgb_prob.predict_proba(X_train)[:, 1]
    test_probs = xgb_prob.predict_proba(X_test)[:, 1]
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(train_probs[y_train == 0], alpha=0.7, bins=30, label='Class 0', color='red')
    plt.hist(train_probs[y_train == 1], alpha=0.7, bins=30, label='Class 1', color='blue')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.title('Training Set Probability Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.hist(test_probs[y_test == 0], alpha=0.7, bins=30, label='Class 0', color='red')
    plt.hist(test_probs[y_test == 1], alpha=0.7, bins=30, label='Class 1', color='blue')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.title('Test Set Probability Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    thresholds = np.linspace(0, 1, 100)
    accuracies = []
    
    for threshold in thresholds:
        pred_classes = (test_probs > threshold).astype(int)
        accuracy = np.mean(pred_classes == y_test)
        accuracies.append(accuracy)
    
    best_threshold_idx = np.argmax(accuracies)
    best_threshold = thresholds[best_threshold_idx]
    best_accuracy = accuracies[best_threshold_idx]
    
    plt.plot(thresholds, accuracies, 'b-', linewidth=2)
    plt.axvline(x=best_threshold, color='r', linestyle='--', 
               label=f'Best threshold: {best_threshold:.3f}')
    plt.axvline(x=0.5, color='g', linestyle='--', alpha=0.7, label='Default: 0.5')
    plt.xlabel('Probability Threshold')
    plt.ylabel('Accuracy')
    plt.title('Threshold vs Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Best threshold: {best_threshold:.3f} (Accuracy: {best_accuracy:.4f})")
    print(f"Default threshold (0.5) accuracy: {xgb_prob.score(X_test, y_test):.4f}")
    
    print("\n=== Model Complexity Analysis ===")
    
    n_estimators_range = [10, 25, 50, 100, 200, 500]
    complexity_results = []
    
    for n_est in n_estimators_range:
        xgb_complex = XGBoost(
            n_estimators=n_est,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        xgb_complex.fit(X_train_sub, y_train_sub, eval_set=(X_val, y_val), verbose=False)
        
        train_acc = xgb_complex.score(X_train_sub, y_train_sub)
        val_acc = xgb_complex.score(X_val, y_val)
        test_acc = xgb_complex.score(X_test, y_test)
        
        complexity_results.append((n_est, train_acc, val_acc, test_acc))
        print(f"n_estimators={n_est}: Train={train_acc:.4f}, Val={val_acc:.4f}, Test={test_acc:.4f}")
    
    n_est_vals = [x[0] for x in complexity_results]
    train_accs_complex = [x[1] for x in complexity_results]
    val_accs_complex = [x[2] for x in complexity_results]
    test_accs_complex = [x[3] for x in complexity_results]
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(n_est_vals, train_accs_complex, 'bo-', label='Train')
    plt.plot(n_est_vals, val_accs_complex, 'ro-', label='Validation')
    plt.plot(n_est_vals, test_accs_complex, 'go-', label='Test')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Accuracy')
    plt.title('Model Complexity vs Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    overfitting_gap = np.array(train_accs_complex) - np.array(val_accs_complex)
    plt.plot(n_est_vals, overfitting_gap, 'mo-', linewidth=2)
    plt.xlabel('Number of Estimators')
    plt.ylabel('Train - Validation Accuracy')
    plt.title('Overfitting Analysis')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    if hasattr(xgb_complex, 'feature_importances_') and xgb_complex.feature_importances_ is not None:
        top_features = np.argsort(xgb_complex.feature_importances_)[-8:]
        plt.barh(range(len(top_features)), xgb_complex.feature_importances_[top_features])
        plt.yticks(range(len(top_features)), [f'Feature {i}' for i in top_features])
        plt.xlabel('Importance')
        plt.title(f'Feature Importances (n_est={n_est})')
    
    plt.subplot(2, 2, 4)
    training_times = [x * 0.1 for x in n_est_vals]
    plt.plot(n_est_vals, training_times, 'co-', linewidth=2)
    plt.xlabel('Number of Estimators')
    plt.ylabel('Training Time (simulated)')
    plt.title('Model Complexity vs Training Time')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    