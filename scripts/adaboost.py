"""AdaBoost classifier using decision stumps as weak learners with example usage and visualization."""

import numpy as np
from sklearn.tree import DecisionTreeClassifier

class AdaBoost:
    def __init__(self, n_estimators=50, learning_rate=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.estimators = []
        self.estimator_weights = []
        self.classes_ = None
        
    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        self.classes_ = np.unique(y)
        n_samples = X.shape[0]
        sample_weights = np.full(n_samples, (1 / n_samples))
        
        for _ in range(self.n_estimators):
            estimator = DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2)
            estimator.fit(X, y, sample_weight=sample_weights)
            predictions = estimator.predict(X)
            incorrect_mask = (predictions != y)
            error = np.sum(sample_weights[incorrect_mask])
            if error == 0:
                error = 1e-10
            estimator_weight = self.learning_rate * np.log((1 - error) / error)
            sample_weights *= np.exp(estimator_weight * incorrect_mask)
            sample_weights /= np.sum(sample_weights)
            self.estimators.append(estimator)
            self.estimator_weights.append(estimator_weight)
            
    def predict(self, X):
        X = np.array(X)
        estimator_preds = np.array([estimator.predict(X) for estimator in self.estimators])
        final_predictions = np.zeros(X.shape[0], dtype=self.classes_.dtype)
        for i, c in enumerate(self.classes_):
            class_votes = np.sum((estimator_preds == c) * self.estimator_weights[:, np.newaxis], axis=0)
            final_predictions[np.argmax(class_votes)] = c
        scores = np.zeros((X.shape[0], len(self.classes_)))
        for i, estimator in enumerate(self.estimators):
            preds = estimator.predict(X)
            for j, class_val in enumerate(self.classes_):
                scores[:, j] += self.estimator_weights[i] * (preds == class_val)
        return self.classes_[np.argmax(scores, axis=1)]

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    X, y = make_classification(n_samples=500, n_features=2, n_informative=2, n_redundant=0,
                               n_clusters_per_class=1, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    adaboost = AdaBoost(n_estimators=50, learning_rate=0.5)
    adaboost.fit(X_train, y_train)
    train_accuracy = adaboost.score(X_train, y_train)
    test_accuracy = adaboost.score(X_test, y_test)
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    def plot_decision_boundary(clf, X, y, title):
        plt.figure(figsize=(10, 7))
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                             np.arange(y_min, y_max, 0.02))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.3)
        plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
        plt.title(title)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()

    plot_decision_boundary(adaboost, X, y, "AdaBoost Decision Boundary")
