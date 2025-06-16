import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class SVM:
    """
    Support Vector Machine implementation from scratch
    Uses Sequential Minimal Optimization (SMO) algorithm for training
    """
    
    def __init__(self, C=1.0, kernel='linear', gamma=1.0, degree=3, coef0=0.0, 
                 tolerance=1e-3, max_iterations=1000):
        """
        Initialize SVM classifier
        
        Parameters:
        C (float): Regularization parameter
        kernel (str): Kernel type ('linear', 'polynomial', 'rbf', 'sigmoid')
        gamma (float): Kernel coefficient for 'rbf', 'polynomial', 'sigmoid'
        degree (int): Degree for polynomial kernel
        coef0 (float): Independent term for polynomial/sigmoid kernels
        tolerance (float): Tolerance for stopping criterion
        max_iterations (int): Maximum number of iterations
        """
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        
        # Model parameters
        self.alphas = None
        self.b = 0
        self.X_support = None
        self.y_support = None
        self.support_indices = None
        self.n_support = 0
        
    def _linear_kernel(self, X1, X2):
        """Linear kernel: K(x1, x2) = x1^T * x2"""
        return np.dot(X1, X2.T)
    
    def _polynomial_kernel(self, X1, X2):
        """Polynomial kernel: K(x1, x2) = (gamma * x1^T * x2 + coef0)^degree"""
        return (self.gamma * np.dot(X1, X2.T) + self.coef0) ** self.degree
    
    def _rbf_kernel(self, X1, X2):
        """RBF (Gaussian) kernel: K(x1, x2) = exp(-gamma * ||x1 - x2||^2)"""
        if X1.ndim == 1:
            X1 = X1.reshape(1, -1)
        if X2.ndim == 1:
            X2 = X2.reshape(1, -1)
        
        # Calculate pairwise squared distances
        X1_norm = np.sum(X1**2, axis=1, keepdims=True)
        X2_norm = np.sum(X2**2, axis=1, keepdims=True)
        distances = X1_norm + X2_norm.T - 2 * np.dot(X1, X2.T)
        
        return np.exp(-self.gamma * distances)
    
    def _sigmoid_kernel(self, X1, X2):
        """Sigmoid kernel: K(x1, x2) = tanh(gamma * x1^T * x2 + coef0)"""
        return np.tanh(self.gamma * np.dot(X1, X2.T) + self.coef0)
    
    def _kernel_function(self, X1, X2):
        """Apply the specified kernel function"""
        if self.kernel == 'linear':
            return self._linear_kernel(X1, X2)
        elif self.kernel == 'polynomial':
            return self._polynomial_kernel(X1, X2)
        elif self.kernel == 'rbf':
            return self._rbf_kernel(X1, X2)
        elif self.kernel == 'sigmoid':
            return self._sigmoid_kernel(X1, X2)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def _compute_error(self, i, X, y, alphas, b):
        """Compute prediction error for sample i"""
        prediction = 0
        for j in range(len(alphas)):
            if alphas[j] > 0:
                prediction += alphas[j] * y[j] * self._kernel_function(X[j:j+1], X[i:i+1])[0, 0]
        prediction += b
        return prediction - y[i]
    
    def _select_second_alpha(self, i, X, y, alphas, b, errors):
        """Select second alpha for SMO algorithm"""
        n_samples = len(y)
        
        # First, try to select alpha that maximizes |E_i - E_j|
        if errors[i] > 0:
            j = np.argmin(errors)
        else:
            j = np.argmax(errors)
        
        if j != i:
            return j
        
        # If that doesn't work, try all non-zero and non-C alphas
        non_bound_indices = np.where((alphas > 0) & (alphas < self.C))[0]
        if len(non_bound_indices) > 1:
            for j in non_bound_indices:
                if j != i:
                    return j
        
        # Finally, try all samples
        for j in range(n_samples):
            if j != i:
                return j
        
        return -1
    
    def _clip_alpha(self, alpha, L, H):
        """Clip alpha to be within bounds [L, H]"""
        if alpha > H:
            return H
        elif alpha < L:
            return L
        else:
            return alpha
    
    def _smo_algorithm(self, X, y):
        """Sequential Minimal Optimization algorithm"""
        n_samples, n_features = X.shape
        alphas = np.zeros(n_samples)
        b = 0
        
        iteration = 0
        passes = 0
        
        while passes < 5 and iteration < self.max_iterations:
            num_changed_alphas = 0
            
            # Calculate errors for all samples
            errors = np.zeros(n_samples)
            for i in range(n_samples):
                errors[i] = self._compute_error(i, X, y, alphas, b)
            
            for i in range(n_samples):
                E_i = errors[i]
                
                # Check KKT conditions
                if ((y[i] * E_i < -self.tolerance and alphas[i] < self.C) or
                    (y[i] * E_i > self.tolerance and alphas[i] > 0)):
                    
                    # Select second alpha
                    j = self._select_second_alpha(i, X, y, alphas, b, errors)
                    if j == -1:
                        continue
                    
                    E_j = errors[j]
                    
                    # Save old alphas
                    alpha_i_old = alphas[i]
                    alpha_j_old = alphas[j]
                    
                    # Compute bounds L and H
                    if y[i] != y[j]:
                        L = max(0, alphas[j] - alphas[i])
                        H = min(self.C, self.C + alphas[j] - alphas[i])
                    else:
                        L = max(0, alphas[i] + alphas[j] - self.C)
                        H = min(self.C, alphas[i] + alphas[j])
                    
                    if L == H:
                        continue
                    
                    # Compute eta
                    eta = (2 * self._kernel_function(X[i:i+1], X[j:j+1])[0, 0] -
                           self._kernel_function(X[i:i+1], X[i:i+1])[0, 0] -
                           self._kernel_function(X[j:j+1], X[j:j+1])[0, 0])
                    
                    if eta >= 0:
                        continue
                    
                    # Compute new alpha_j
                    alphas[j] = alphas[j] - (y[j] * (E_i - E_j)) / eta
                    alphas[j] = self._clip_alpha(alphas[j], L, H)
                    
                    if abs(alphas[j] - alpha_j_old) < 1e-5:
                        continue
                    
                    # Compute new alpha_i
                    alphas[i] = alphas[i] + y[i] * y[j] * (alpha_j_old - alphas[j])
                    
                    # Compute new threshold b
                    b1 = (b - E_i - y[i] * (alphas[i] - alpha_i_old) * 
                          self._kernel_function(X[i:i+1], X[i:i+1])[0, 0] -
                          y[j] * (alphas[j] - alpha_j_old) * 
                          self._kernel_function(X[i:i+1], X[j:j+1])[0, 0])
                    
                    b2 = (b - E_j - y[i] * (alphas[i] - alpha_i_old) * 
                          self._kernel_function(X[i:i+1], X[j:j+1])[0, 0] -
                          y[j] * (alphas[j] - alpha_j_old) * 
                          self._kernel_function(X[j:j+1], X[j:j+1])[0, 0])
                    
                    if 0 < alphas[i] < self.C:
                        b = b1
                    elif 0 < alphas[j] < self.C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2
                    
                    num_changed_alphas += 1
            
            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0
            
            iteration += 1
        
        return alphas, b
    
    def fit(self, X, y):
        """
        Train the SVM classifier
        
        Parameters:
        X (array): Training features (m x n)
        y (array): Training labels (m,) - should be -1 or 1
        """
        X = np.array(X)
        y = np.array(y)
        
        # Convert labels to -1, 1 if they are 0, 1
        if np.all(np.unique(y) == [0, 1]):
            y = 2 * y - 1
        elif not np.all(np.isin(y, [-1, 1])):
            raise ValueError("Labels must be binary (0/1 or -1/1)")
        
        # Train using SMO algorithm
        self.alphas, self.b = self._smo_algorithm(X, y)
        
        # Store support vectors
        support_mask = self.alphas > 1e-5
        self.support_indices = np.where(support_mask)[0]
        self.X_support = X[support_mask]
        self.y_support = y[support_mask]
        self.alphas = self.alphas[support_mask]
        self.n_support = len(self.X_support)
        
        print(f"SVM trained with {self.n_support} support vectors")
    
    def _decision_function(self, X):
        """Compute the decision function"""
        if self.X_support is None:
            raise ValueError("Model must be fitted before making predictions")
        
        X = np.array(X)
        decision = np.zeros(X.shape[0])
        
        for i in range(len(self.alphas)):
            decision += (self.alphas[i] * self.y_support[i] * 
                        self._kernel_function(self.X_support[i:i+1], X).ravel())
        
        return decision + self.b
    
    def predict(self, X):
        """
        Make predictions on new data
        
        Parameters:
        X (array): Features to predict on
        
        Returns:
        array: Predicted class labels (-1 or 1)
        """
        decision = self._decision_function(X)
        return np.sign(decision)
    
    def predict_proba(self, X):
        """
        Predict class probabilities using Platt scaling (simplified)
        
        Parameters:
        X (array): Features to predict on
        
        Returns:
        array: Class probabilities
        """
        decision = self._decision_function(X)
        # Simple sigmoid transformation (not true Platt scaling)
        probabilities = 1 / (1 + np.exp(-decision))
        return np.column_stack([1 - probabilities, probabilities])
    
    def score(self, X, y):
        """
        Calculate accuracy score
        
        Parameters:
        X (array): Features
        y (array): True labels
        
        Returns:
        float: Accuracy score
        """
        # Convert labels to -1, 1 if they are 0, 1
        if np.all(np.unique(y) == [0, 1]):
            y = 2 * y - 1
        
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def plot_decision_boundary(self, X, y, resolution=0.02):
        """Plot decision boundary for 2D data"""
        if X.shape[1] != 2:
            print("Decision boundary plot only available for 2D data")
            return
        
        # Convert labels for plotting
        y_plot = y.copy()
        if np.all(np.unique(y_plot) == [-1, 1]):
            y_plot = (y_plot + 1) / 2  # Convert -1,1 to 0,1
        
        plt.figure(figsize=(12, 8))
        
        # Plot data points
        colors = ['red', 'blue']
        labels = ['Class 0', 'Class 1']
        
        for i, label in enumerate(np.unique(y_plot)):
            mask = y_plot == label
            plt.scatter(X[mask, 0], X[mask, 1], c=colors[i], 
                       alpha=0.7, s=50, label=labels[i])
        
        # Highlight support vectors
        if self.X_support is not None:
            plt.scatter(self.X_support[:, 0], self.X_support[:, 1], 
                       s=100, facecolors='none', edgecolors='black', 
                       linewidth=2, label='Support Vectors')
        
        # Create mesh for decision boundary
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                            np.arange(y_min, y_max, resolution))
        
        # Make predictions on mesh
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = self._decision_function(mesh_points)
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary and margins
        plt.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['gray', 'black', 'gray'],
                   linestyles=['--', '-', '--'], linewidths=[1, 2, 1])
        plt.contourf(xx, yy, Z, levels=50, alpha=0.3, cmap='RdBu')
        
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(f'SVM Decision Boundary (C={self.C}, kernel={self.kernel})')
        plt.legend()
        plt.colorbar(label='Decision Function')
        plt.show()
    
    def plot_support_vectors(self, X, y):
        """Plot support vectors with their alpha values"""
        if self.X_support is None:
            print("Model must be fitted first")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Convert labels for plotting
        y_plot = y.copy()
        if np.all(np.unique(y_plot) == [-1, 1]):
            y_plot = (y_plot + 1) / 2
        
        # Plot all points
        colors = ['red', 'blue']
        for i, label in enumerate(np.unique(y_plot)):
            mask = y_plot == label
            plt.scatter(X[mask, 0], X[mask, 1], c=colors[i], 
                       alpha=0.3, s=30, label=f'Class {i}')
        
        # Plot support vectors with size proportional to alpha
        if X.shape[1] == 2:
            y_support_plot = self.y_support.copy()
            if np.all(np.unique(y_support_plot) == [-1, 1]):
                y_support_plot = (y_support_plot + 1) / 2
            
            for i, (sv, alpha, label) in enumerate(zip(self.X_support, self.alphas, y_support_plot)):
                plt.scatter(sv[0], sv[1], c=colors[int(label)], 
                           s=200 * alpha / np.max(self.alphas), 
                           edgecolors='black', linewidth=2,
                           label='Support Vector' if i == 0 else "")
        
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(f'Support Vectors (size ∝ alpha value)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    from sklearn.datasets import make_classification, make_circles
    
    print("=== Linear SVM Example ===")
    
    # Linearly separable data
    X_linear, y_linear = make_classification(n_samples=200, n_features=2, 
                                           n_redundant=0, n_informative=2,
                                           n_clusters_per_class=1, random_state=42)
    
    # Split data
    split_idx = int(0.8 * len(X_linear))
    X_train, X_test = X_linear[:split_idx], X_linear[split_idx:]
    y_train, y_test = y_linear[:split_idx], y_linear[split_idx:]
    
    # Train linear SVM
    svm_linear = SVM(C=1.0, kernel='linear')
    svm_linear.fit(X_train, y_train)
    
    # Evaluate
    train_accuracy = svm_linear.score(X_train, y_train)
    test_accuracy = svm_linear.score(X_test, y_test)
    
    print(f"Linear SVM Train Accuracy: {train_accuracy:.4f}")
    print(f"Linear SVM Test Accuracy: {test_accuracy:.4f}")
    print(f"Number of Support Vectors: {svm_linear.n_support}")
    
    # Plot results
    svm_linear.plot_decision_boundary(X_train, y_train)
    svm_linear.plot_support_vectors(X_train, y_train)
    
    print("\n=== RBF SVM Example ===")
    
    # Non-linearly separable data (circles)
    X_circles, y_circles = make_circles(n_samples=200, noise=0.1, factor=0.3, random_state=42)
    
    # Split data
    split_idx = int(0.8 * len(X_circles))
    X_train_c, X_test_c = X_circles[:split_idx], X_circles[split_idx:]
    y_train_c, y_test_c = y_circles[:split_idx], y_circles[split_idx:]
    
    # Train RBF SVM
    svm_rbf = SVM(C=1.0, kernel='rbf', gamma=1.0)
    svm_rbf.fit(X_train_c, y_train_c)
    
    # Evaluate
    train_accuracy_rbf = svm_rbf.score(X_train_c, y_train_c)
    test_accuracy_rbf = svm_rbf.score(X_test_c, y_test_c)
    
    print(f"RBF SVM Train Accuracy: {train_accuracy_rbf:.4f}")
    print(f"RBF SVM Test Accuracy: {test_accuracy_rbf:.4f}")
    print(f"Number of Support Vectors: {svm_rbf.n_support}")
    
    # Plot results
    svm_rbf.plot_decision_boundary(X_train_c, y_train_c)
    
    print("\n=== Comparing Different Kernels ===")
    
    kernels = ['linear', 'polynomial', 'rbf']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, kernel in enumerate(kernels):
        # Train SVM with different kernel
        if kernel == 'polynomial':
            svm_comp = SVM(C=1.0, kernel=kernel, degree=3, gamma=1.0)
        else:
            svm_comp = SVM(C=1.0, kernel=kernel, gamma=1.0)
        
        svm_comp.fit(X_train_c, y_train_c)
        accuracy = svm_comp.score(X_test_c, y_test_c)
        
        # Plot decision boundary
        plt.subplot(1, 3, i+1)
        
        # Plot data points
        colors = ['red', 'blue']
        for j, label in enumerate(np.unique(y_train_c)):
            mask = y_train_c == label
            plt.scatter(X_train_c[mask, 0], X_train_c[mask, 1], 
                       c=colors[j], alpha=0.7, s=50, label=f'Class {j}')
        
        # Plot support vectors
        if svm_comp.X_support is not None:
            plt.scatter(svm_comp.X_support[:, 0], svm_comp.X_support[:, 1], 
                       s=100, facecolors='none', edgecolors='black', 
                       linewidth=2, label='Support Vectors')
        
        # Create mesh and plot decision boundary
        x_min, x_max = X_train_c[:, 0].min() - 1, X_train_c[:, 0].max() + 1
        y_min, y_max = X_train_c[:, 1].min() - 1, X_train_c[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                            np.arange(y_min, y_max, 0.02))
        
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = svm_comp._decision_function(mesh_points)
        Z = Z.reshape(xx.shape)
        
        plt.contour(xx, yy, Z, levels=[0], colors=['black'], linewidths=[2])
        plt.contourf(xx, yy, Z, levels=50, alpha=0.3, cmap='RdBu')
        
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(f'{kernel.title()} Kernel\nAccuracy: {accuracy:.3f}, SVs: {svm_comp.n_support}')
        plt.legend()
    
    plt.tight_layout()
    plt.show()