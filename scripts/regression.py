import numpy as np
import matplotlib.pyplot as plt

class RegressionModel:
    def __init__(self, degree=1, method='closed_form', learning_rate=0.01, iterations=1000):
        self.degree = degree
        self.method = method
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.coefficients = None

    def _polynomial_features(self, X):
        X_poly = np.ones((X.shape[0], 1))
        for d in range(1, self.degree + 1):
            X_poly = np.hstack((X_poly, np.power(X, d)))
        return X_poly

    def fit(self, X, y):
        X_b = self._polynomial_features(X)
        if self.method == 'closed_form':
            self.coefficients = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        elif self.method == 'gradient_descent':
            self.coefficients = np.zeros(X_b.shape[1])
            for _ in range(self.iterations):
                gradients = (-2 / X_b.shape[0]) * X_b.T.dot(X_b.dot(self.coefficients) - y)
                self.coefficients -= self.learning_rate * gradients

    def predict(self, X):
        X_b = self._polynomial_features(X)
        print(X_b)
        return X_b.dot(self.coefficients)

# Example usage:
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(0)
    X = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 100)

    # Initialize and fit the model
    degree = 2  # Change this to 1 for linear, 2 for quadratic, etc.
    method = 'gradient_descent'  # Change to 'closed_form' for closed-form solution
    model = RegressionModel(degree=degree, method=method)
    model.fit(X, y)

    # Predict
    X_test = np.linspace(0, 5, 100).reshape(-1, 1)
    y_pred = model.predict(X_test)

    print(y_pred)

    # Plot results
    plt.scatter(X, y, color='blue', label='Data points')
    plt.plot(X_test, y_pred, color='red', label=f'Polynomial degree {degree} ({method})')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.show()
