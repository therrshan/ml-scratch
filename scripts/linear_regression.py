"""
Linear Regression using gradient descent with methods for training, prediction, scoring (R²), 
and cost visualization. 
Example loads data, fits model, evaluates, and plots results.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class LinearRegression:
    
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.cost_history = []
        
    def _add_bias(self, X):
        return np.c_[np.ones(X.shape[0]), X]
    
    def _compute_cost(self, X, y):
        m = X.shape[0]
        predictions = X.dot(self.theta)
        cost = (1/(2*m)) * np.sum((predictions - y)**2)
        return cost
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        X_with_bias = self._add_bias(X)
        n_features = X_with_bias.shape[1]
        self.theta = np.random.normal(0, 1, n_features).reshape(-1, 1)
        m = X.shape[0]
        prev_cost = float('inf')

        for _ in range(self.max_iterations):
            predictions = X_with_bias.dot(self.theta)
            cost = self._compute_cost(X_with_bias, y)
            self.cost_history.append(cost)
            gradients = (1/m) * X_with_bias.T.dot(predictions - y)
            self.theta -= self.learning_rate * gradients
            if abs(prev_cost - cost) < self.tolerance:
                break
            prev_cost = cost

        self.bias = self.theta[0]
        self.weights = self.theta[1:]
        
    def predict(self, X):
        X = np.array(X)
        X_with_bias = self._add_bias(X)
        return X_with_bias.dot(self.theta)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        return 1 - (ss_res / ss_tot)
    
    def plot_cost_history(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history)
        plt.title('Cost Function Over Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    np.random.seed(42)

    X = 2 * np.random.rand(100, 1)
    y = 3 * X.ravel() + 2 + np.random.randn(100) * 0.5
    
    model = LinearRegression(learning_rate=0.1, max_iterations=1000)
    model.fit(X, y)
    
    predictions = model.predict(X)
    
    print(f"Weights: {model.weights.ravel()}")
    print(f"Bias: {model.bias[0]}")
    print(f"R-squared: {model.score(X, y):.4f}")
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X, y, alpha=0.6, label='Data')
    plt.plot(X, predictions, 'r-', label='Predictions')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.title('Linear Regression Fit')
    
    plt.subplot(1, 2, 2)
    plt.plot(model.cost_history)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost History')
    
    plt.tight_layout()
    plt.show()
