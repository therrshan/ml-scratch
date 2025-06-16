import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    """
    Linear Regression implementation from scratch using Gradient Descent
    """
    
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        """
        Initialize Linear Regression model
        
        Parameters:
        learning_rate (float): Step size for gradient descent
        max_iterations (int): Maximum number of iterations
        tolerance (float): Convergence tolerance
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.cost_history = []
        
    def _add_bias(self, X):
        """Add bias column to feature matrix"""
        return np.c_[np.ones(X.shape[0]), X]
    
    def _compute_cost(self, X, y):
        """Compute Mean Squared Error cost"""
        m = X.shape[0]
        predictions = X.dot(self.theta)
        cost = (1/(2*m)) * np.sum((predictions - y)**2)
        return cost
    
    def fit(self, X, y):
        """
        Train the linear regression model
        
        Parameters:
        X (array): Training features (m x n)
        y (array): Training targets (m,)
        """
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Add bias term
        X_with_bias = self._add_bias(X)
        
        # Initialize parameters
        n_features = X_with_bias.shape[1]
        self.theta = np.random.normal(0, 0.01, n_features)
        
        # Gradient Descent
        m = X.shape[0]
        prev_cost = float('inf')
        
        for i in range(self.max_iterations):
            # Forward pass
            predictions = X_with_bias.dot(self.theta)
            
            # Compute cost
            cost = self._compute_cost(X_with_bias, y)
            self.cost_history.append(cost)
            
            # Compute gradients
            gradients = (1/m) * X_with_bias.T.dot(predictions - y)
            
            # Update parameters
            self.theta -= self.learning_rate * gradients
            
            # Check convergence
            if abs(prev_cost - cost) < self.tolerance:
                print(f"Converged after {i+1} iterations")
                break
            prev_cost = cost
        
        # Extract weights and bias
        self.bias = self.theta[0]
        self.weights = self.theta[1:]
        
    def predict(self, X):
        """
        Make predictions on new data
        
        Parameters:
        X (array): Features to predict on
        
        Returns:
        array: Predictions
        """
        X = np.array(X)
        X_with_bias = self._add_bias(X)
        return X_with_bias.dot(self.theta)
    
    def score(self, X, y):
        """
        Calculate R-squared score
        
        Parameters:
        X (array): Features
        y (array): True targets
        
        Returns:
        float: R-squared score
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        return 1 - (ss_res / ss_tot)
    
    def plot_cost_history(self):
        """Plot the cost function over iterations"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history)
        plt.title('Cost Function Over Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.grid(True)
        plt.show()

# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(100, 1)
    y = 2 * X.ravel() + 1 + np.random.randn(100) * 0.1
    
    # Create and train model
    model = LinearRegression(learning_rate=0.1, max_iterations=1000)
    model.fit(X, y)
    
    # Make predictions
    predictions = model.predict(X)
    
    # Print results
    print(f"Weights: {model.weights}")
    print(f"Bias: {model.bias}")
    print(f"R-squared: {model.score(X, y):.4f}")
    
    # Plot results
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