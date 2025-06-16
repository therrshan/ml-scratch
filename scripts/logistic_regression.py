import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    """
    Logistic Regression implementation from scratch using Gradient Descent
    """
    
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        """
        Initialize Logistic Regression model
        
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
    
    def _sigmoid(self, z):
        """Sigmoid activation function"""
        # Clip z to prevent overflow
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))
    
    def _compute_cost(self, X, y):
        """Compute logistic regression cost (cross-entropy)"""
        m = X.shape[0]
        z = X.dot(self.theta)
        predictions = self._sigmoid(z)
        
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        
        cost = -(1/m) * np.sum(y * np.log(predictions) + (1-y) * np.log(1-predictions))
        return cost
    
    def fit(self, X, y):
        """
        Train the logistic regression model
        
        Parameters:
        X (array): Training features (m x n)
        y (array): Training targets (m,) - binary (0 or 1)
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
            z = X_with_bias.dot(self.theta)
            predictions = self._sigmoid(z)
            
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
    
    def predict_proba(self, X):
        """
        Predict class probabilities
        
        Parameters:
        X (array): Features to predict on
        
        Returns:
        array: Predicted probabilities
        """
        X = np.array(X)
        X_with_bias = self._add_bias(X)
        z = X_with_bias.dot(self.theta)
        return self._sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        """
        Make binary predictions
        
        Parameters:
        X (array): Features to predict on
        threshold (float): Decision threshold
        
        Returns:
        array: Binary predictions (0 or 1)
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
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
    
    def plot_cost_history(self):
        """Plot the cost function over iterations"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history)
        plt.title('Cost Function Over Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.grid(True)
        plt.show()
    
    def plot_decision_boundary(self, X, y):
        """Plot decision boundary for 2D data"""
        if X.shape[1] != 2:
            print("Decision boundary plot only available for 2D data")
            return
        
        plt.figure(figsize=(10, 8))
        
        # Plot data points
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter)
        
        # Create a mesh to plot the decision boundary
        h = 0.01
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # Make predictions on the mesh
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = self.predict_proba(mesh_points)
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        plt.contour(xx, yy, Z, levels=[0.5], colors='red', linestyles='--', linewidths=2)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Logistic Regression Decision Boundary')
        plt.show()

# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                             n_informative=2, n_clusters_per_class=1, random_state=42)
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Create and train model
    model = LogisticRegression(learning_rate=0.1, max_iterations=1000)
    model.fit(X_train, y_train)
    
    # Make predictions
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    
    # Print results
    print(f"Weights: {model.weights}")
    print(f"Bias: {model.bias}")
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(model.cost_history)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost History')
    
    plt.subplot(1, 3, 2)
    model.plot_decision_boundary(X_train, y_train)
    plt.title('Training Data Decision Boundary')
    
    plt.subplot(1, 3, 3)
    model.plot_decision_boundary(X_test, y_test)
    plt.title('Test Data Decision Boundary')
    
    plt.tight_layout()
    plt.show()