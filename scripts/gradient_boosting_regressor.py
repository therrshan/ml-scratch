"""
Gradient Boosting Regressor implemented from scratch using decision trees
as weak learners and mean initialization.
"""

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyRegressor

class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.estimators_ = []
        self.initial_estimator_ = None

    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        self.initial_estimator_ = DummyRegressor(strategy='mean')
        self.initial_estimator_.fit(X, y)
        current_predictions = self.initial_estimator_.predict(X)
        for _ in range(self.n_estimators):
            residuals = y - current_predictions
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=42
            )
            tree.fit(X, residuals)
            update = self.learning_rate * tree.predict(X)
            current_predictions += update
            self.estimators_.append(tree)

    def predict(self, X):
        X = np.array(X)
        predictions = self.initial_estimator_.predict(X)
        for tree in self.estimators_:
            predictions += self.learning_rate * tree.predict(X)
        return predictions

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return 1 - (ss_res / ss_tot)

if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    X, y = make_regression(n_samples=500, n_features=1, noise=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
    gbr.fit(X_train, y_train)
    train_r2 = gbr.score(X_train, y_train)
    test_r2 = gbr.score(X_test, y_test)
    print(f"Train R-squared: {train_r2:.4f}")
    print(f"Test R-squared: {test_r2:.4f}")
    y_pred = gbr.predict(X_test)
    plt.figure(figsize=(10, 7))
    plt.scatter(X_test, y_test, color='black', alpha=0.5, label='Actual values')
    sort_axis = np.argsort(X_test.ravel())
    plt.plot(X_test[sort_axis], y_pred[sort_axis], color='red', linewidth=2, label='Predicted values')
    plt.title("Gradient Boosting Regressor")
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.legend()
    plt.show()
