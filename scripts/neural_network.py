import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    """
    Multi-layer Neural Network implementation from scratch
    Supports multiple hidden layers with customizable activation functions
    """
    
    def __init__(self, layers, learning_rate=0.01, activation='relu', 
                 output_activation='softmax', max_iterations=1000, 
                 tolerance=1e-6, batch_size=32, regularization=None, lambda_reg=0.01):
        """
        Initialize Neural Network
        
        Parameters:
        layers (list): List of layer sizes [input_size, hidden1, hidden2, ..., output_size]
        learning_rate (float): Learning rate for gradient descent
        activation (str): Activation function for hidden layers ('relu', 'sigmoid', 'tanh')
        output_activation (str): Activation function for output layer ('softmax', 'sigmoid', 'linear')
        max_iterations (int): Maximum number of training iterations
        tolerance (float): Convergence tolerance
        batch_size (int): Batch size for mini-batch gradient descent
        regularization (str): Regularization type ('l1', 'l2', None)
        lambda_reg (float): Regularization strength
        """
        self.layers = layers
        self.learning_rate = learning_rate
        self.activation = activation
        self.output_activation = output_activation
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.batch_size = batch_size
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        self._initialize_parameters()
        
        # Training history
        self.cost_history = []
        self.accuracy_history = []
        
    def _initialize_parameters(self):
        """Initialize weights and biases using Xavier initialization"""
        np.random.seed(42)
        
        for i in range(len(self.layers) - 1):
            # Xavier initialization
            w = np.random.randn(self.layers[i], self.layers[i+1]) * np.sqrt(2.0 / self.layers[i])
            b = np.zeros((1, self.layers[i+1]))
            
            self.weights.append(w)
            self.biases.append(b)
    
    def _relu(self, z):
        """ReLU activation function"""
        return np.maximum(0, z)
    
    def _relu_derivative(self, z):
        """Derivative of ReLU"""
        return (z > 0).astype(float)
    
    def _sigmoid(self, z):
        """Sigmoid activation function"""
        # Clip z to prevent overflow
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))
    
    def _sigmoid_derivative(self, z):
        """Derivative of sigmoid"""
        s = self._sigmoid(z)
        return s * (1 - s)
    
    def _tanh(self, z):
        """Tanh activation function"""
        return np.tanh(z)
    
    def _tanh_derivative(self, z):
        """Derivative of tanh"""
        return 1 - np.tanh(z)**2
    
    def _softmax(self, z):
        """Softmax activation function"""
        # Subtract max for numerical stability
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def _linear(self, z):
        """Linear activation function"""
        return z
    
    def _apply_activation(self, z, activation):
        """Apply specified activation function"""
        if activation == 'relu':
            return self._relu(z)
        elif activation == 'sigmoid':
            return self._sigmoid(z)
        elif activation == 'tanh':
            return self._tanh(z)
        elif activation == 'softmax':
            return self._softmax(z)
        elif activation == 'linear':
            return self._linear(z)
        else:
            raise ValueError(f"Unknown activation function: {activation}")
    
    def _apply_activation_derivative(self, z, activation):
        """Apply derivative of specified activation function"""
        if activation == 'relu':
            return self._relu_derivative(z)
        elif activation == 'sigmoid':
            return self._sigmoid_derivative(z)
        elif activation == 'tanh':
            return self._tanh_derivative(z)
        else:
            raise ValueError(f"Derivative not implemented for: {activation}")
    
    def _forward_propagation(self, X):
        """Forward propagation through the network"""
        activations = [X]
        z_values = []
        
        # Forward through hidden layers
        for i in range(len(self.weights) - 1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            a = self._apply_activation(z, self.activation)
            
            z_values.append(z)
            activations.append(a)
        
        # Forward through output layer
        z_output = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        a_output = self._apply_activation(z_output, self.output_activation)
        
        z_values.append(z_output)
        activations.append(a_output)
        
        return activations, z_values
    
    def _compute_cost(self, y_true, y_pred):
        """Compute cost function"""
        m = y_true.shape[0]
        
        if self.output_activation == 'softmax':
            # Cross-entropy loss for multi-class classification
            # Add small epsilon to prevent log(0)
            epsilon = 1e-15
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            cost = -np.sum(y_true * np.log(y_pred)) / m
        elif self.output_activation == 'sigmoid':
            # Binary cross-entropy loss
            epsilon = 1e-15
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            cost = -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / m
        else:
            # Mean squared error for regression
            cost = np.sum((y_true - y_pred)**2) / (2 * m)
        
        # Add regularization
        if self.regularization == 'l2':
            l2_cost = sum(np.sum(w**2) for w in self.weights)
            cost += self.lambda_reg * l2_cost / (2 * m)
        elif self.regularization == 'l1':
            l1_cost = sum(np.sum(np.abs(w)) for w in self.weights)
            cost += self.lambda_reg * l1_cost / m
        
        return cost
    
    def _backward_propagation(self, X, y, activations, z_values):
        """Backward propagation to compute gradients"""
        m = X.shape[0]
        n_layers = len(self.weights)
        
        # Initialize gradients
        dW = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]
        
        # Output layer error
        if self.output_activation == 'softmax':
            # For softmax with cross-entropy, the gradient simplifies
            dz = activations[-1] - y
        elif self.output_activation == 'sigmoid':
            dz = (activations[-1] - y) * self._sigmoid_derivative(z_values[-1])
        else:
            # For linear output (regression)
            dz = activations[-1] - y
            # Ensure proper shape for regression output
            if len(dz.shape) == 1:
                dz = dz.reshape(-1, 1)
        
        # Gradients for output layer
        dW[-1] = np.dot(activations[-2].T, dz) / m
        db[-1] = np.sum(dz, axis=0, keepdims=True) / m
        
        # Add regularization to weights
        if self.regularization == 'l2':
            dW[-1] += self.lambda_reg * self.weights[-1] / m
        elif self.regularization == 'l1':
            dW[-1] += self.lambda_reg * np.sign(self.weights[-1]) / m
        
        # Backpropagate through hidden layers
        for i in range(n_layers - 2, -1, -1):
            # Compute error for current layer
            dz = np.dot(dz, self.weights[i+1].T) * self._apply_activation_derivative(z_values[i], self.activation)
            
            # Compute gradients
            dW[i] = np.dot(activations[i].T, dz) / m
            db[i] = np.sum(dz, axis=0, keepdims=True) / m
            
            # Add regularization
            if self.regularization == 'l2':
                dW[i] += self.lambda_reg * self.weights[i] / m
            elif self.regularization == 'l1':
                dW[i] += self.lambda_reg * np.sign(self.weights[i]) / m
        
        return dW, db
    
    def _create_mini_batches(self, X, y):
        """Create mini-batches for training"""
        m = X.shape[0]
        mini_batches = []
        
        # Shuffle the data
        permutation = np.random.permutation(m)
        X_shuffled = X[permutation]
        y_shuffled = y[permutation]
        
        # Create mini-batches
        num_complete_batches = m // self.batch_size
        
        for k in range(num_complete_batches):
            start = k * self.batch_size
            end = start + self.batch_size
            mini_batch_X = X_shuffled[start:end]
            mini_batch_y = y_shuffled[start:end]
            mini_batches.append((mini_batch_X, mini_batch_y))
        
        # Handle remaining samples
        if m % self.batch_size != 0:
            start = num_complete_batches * self.batch_size
            mini_batch_X = X_shuffled[start:]
            mini_batch_y = y_shuffled[start:]
            mini_batches.append((mini_batch_X, mini_batch_y))
        
        return mini_batches
    
    def _one_hot_encode(self, y, num_classes):
        """Convert labels to one-hot encoding"""
        one_hot = np.zeros((len(y), num_classes))
        one_hot[np.arange(len(y)), y] = 1
        return one_hot
    
    def fit(self, X, y):
        """
        Train the neural network
        
        Parameters:
        X (array): Training features (m x n)
        y (array): Training labels (m,) or (m, num_classes)
        """
        X = np.array(X)
        y = np.array(y)
        
        # Prepare labels
        if len(y.shape) == 1:
            if self.output_activation == 'softmax':
                # Multi-class classification - convert to one-hot
                num_classes = len(np.unique(y))
                y = self._one_hot_encode(y, num_classes)
            elif self.output_activation == 'sigmoid':
                # Binary classification - reshape
                y = y.reshape(-1, 1)
            elif self.output_activation == 'linear':
                # Regression - reshape to column vector
                y = y.reshape(-1, 1)
        
        prev_cost = float('inf')
        
        for iteration in range(self.max_iterations):
            epoch_cost = 0
            epoch_accuracy = 0
            num_batches = 0
            
            # Create mini-batches
            mini_batches = self._create_mini_batches(X, y)
            
            for mini_batch_X, mini_batch_y in mini_batches:
                # Forward propagation
                activations, z_values = self._forward_propagation(mini_batch_X)
                
                # Compute cost
                cost = self._compute_cost(mini_batch_y, activations[-1])
                epoch_cost += cost
                
                # Compute accuracy
                if self.output_activation == 'softmax':
                    predictions = np.argmax(activations[-1], axis=1)
                    true_labels = np.argmax(mini_batch_y, axis=1)
                    accuracy = np.mean(predictions == true_labels)
                elif self.output_activation == 'sigmoid':
                    predictions = (activations[-1] > 0.5).astype(int)
                    accuracy = np.mean(predictions == mini_batch_y)
                else:
                    # For regression, use R-squared
                    ss_res = np.sum((mini_batch_y - activations[-1])**2)
                    ss_tot = np.sum((mini_batch_y - np.mean(mini_batch_y))**2)
                    accuracy = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                
                epoch_accuracy += accuracy
                num_batches += 1
                
                # Backward propagation
                dW, db = self._backward_propagation(mini_batch_X, mini_batch_y, activations, z_values)
                
                # Update parameters
                for i in range(len(self.weights)):
                    self.weights[i] -= self.learning_rate * dW[i]
                    self.biases[i] -= self.learning_rate * db[i]
            
            # Record average cost and accuracy for the epoch
            avg_cost = epoch_cost / num_batches
            avg_accuracy = epoch_accuracy / num_batches
            
            self.cost_history.append(avg_cost)
            self.accuracy_history.append(avg_accuracy)
            
            # Print progress
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Cost = {avg_cost:.6f}, Accuracy = {avg_accuracy:.4f}")
            
            # Check convergence
            if abs(prev_cost - avg_cost) < self.tolerance:
                print(f"Converged after {iteration + 1} iterations")
                break
            
            prev_cost = avg_cost
        
        print(f"Training completed. Final cost: {self.cost_history[-1]:.6f}")
    
    def predict(self, X):
        """
        Make predictions on new data
        
        Parameters:
        X (array): Features to predict on
        
        Returns:
        array: Predictions
        """
        X = np.array(X)
        activations, _ = self._forward_propagation(X)
        
        if self.output_activation == 'softmax':
            return np.argmax(activations[-1], axis=1)
        elif self.output_activation == 'sigmoid':
            return (activations[-1] > 0.5).astype(int).ravel()
        else:
            return activations[-1].ravel()
    
    def predict_proba(self, X):
        """
        Predict class probabilities
        
        Parameters:
        X (array): Features to predict on
        
        Returns:
        array: Class probabilities
        """
        X = np.array(X)
        activations, _ = self._forward_propagation(X)
        return activations[-1]
    
    def score(self, X, y):
        """
        Calculate accuracy score for classification or R-squared for regression
        
        Parameters:
        X (array): Features
        y (array): True labels/values
        
        Returns:
        float: Score
        """
        if self.output_activation in ['softmax', 'sigmoid']:
            predictions = self.predict(X)
            return np.mean(predictions == y)
        else:
            predictions = self.predict(X)
            ss_res = np.sum((y - predictions)**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    def plot_training_history(self):
        """Plot training cost and accuracy history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot cost
        ax1.plot(self.cost_history)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Cost')
        ax1.set_title('Training Cost')
        ax1.grid(True, alpha=0.3)
        
        # Plot accuracy
        ax2.plot(self.accuracy_history)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training Accuracy')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_decision_boundary(self, X, y, resolution=0.02):
        """Plot decision boundary for 2D classification data"""
        if X.shape[1] != 2:
            print("Decision boundary plot only available for 2D data")
            return
        
        if self.output_activation not in ['softmax', 'sigmoid']:
            print("Decision boundary plot only available for classification")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Plot data points
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        unique_labels = np.unique(y)
        
        for i, label in enumerate(unique_labels):
            mask = y == label
            plt.scatter(X[mask, 0], X[mask, 1], 
                       c=colors[i % len(colors)], alpha=0.7, s=50, 
                       label=f'Class {label}')
        
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
        plt.title(f'Neural Network Decision Boundary\nArchitecture: {self.layers}')
        plt.legend()
        plt.colorbar()
        plt.show()
    
    def visualize_weights(self, layer_idx=0):
        """
        Visualize weights of a specific layer
        
        Parameters:
        layer_idx (int): Index of layer to visualize
        """
        if layer_idx >= len(self.weights):
            print(f"Layer index {layer_idx} out of range. Network has {len(self.weights)} layers.")
            return
        
        weights = self.weights[layer_idx]
        
        plt.figure(figsize=(12, 8))
        
        # Plot weight matrix as heatmap
        plt.subplot(2, 2, 1)
        plt.imshow(weights, cmap='RdBu', aspect='auto')
        plt.colorbar()
        plt.title(f'Layer {layer_idx} Weights Heatmap')
        plt.xlabel('Neurons in Next Layer')
        plt.ylabel('Neurons in Current Layer')
        
        # Plot weight distribution
        plt.subplot(2, 2, 2)
        plt.hist(weights.flatten(), bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Weight Value')
        plt.ylabel('Frequency')
        plt.title(f'Layer {layer_idx} Weight Distribution')
        plt.grid(True, alpha=0.3)
        
        # Plot weight statistics
        plt.subplot(2, 2, 3)
        stats = {
            'Mean': np.mean(weights),
            'Std': np.std(weights),
            'Min': np.min(weights),
            'Max': np.max(weights),
            'Median': np.median(weights)
        }
        
        plt.bar(stats.keys(), stats.values(), alpha=0.7)
        plt.title(f'Layer {layer_idx} Weight Statistics')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Plot bias values if they exist
        plt.subplot(2, 2, 4)
        biases = self.biases[layer_idx].flatten()
        plt.bar(range(len(biases)), biases, alpha=0.7)
        plt.xlabel('Neuron Index')
        plt.ylabel('Bias Value')
        plt.title(f'Layer {layer_idx} Bias Values')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def get_activations(self, X, layer_idx=None):
        """
        Get activations for a specific layer
        
        Parameters:
        X (array): Input data
        layer_idx (int): Layer index (None for all layers)
        
        Returns:
        array or list: Activations
        """
        activations, _ = self._forward_propagation(X)
        
        if layer_idx is None:
            return activations
        elif layer_idx < len(activations):
            return activations[layer_idx]
        else:
            raise ValueError(f"Layer index {layer_idx} out of range")

# Example usage
if __name__ == "__main__":
    print("=== Neural Network Classification Example ===")
    
    # Generate sample data
    np.random.seed(42)
    from sklearn.datasets import make_classification, make_circles, make_moons
    
    # Multi-class classification
    X_class, y_class = make_classification(n_samples=1000, n_features=4, 
                                         n_informative=4, n_redundant=0,
                                         n_classes=3, n_clusters_per_class=1, 
                                         random_state=42)
    
    # Split data
    split_idx = int(0.8 * len(X_class))
    X_train, X_test = X_class[:split_idx], X_class[split_idx:]
    y_train, y_test = y_class[:split_idx], y_class[split_idx:]
    
    # Create and train neural network
    nn_classifier = NeuralNetwork(
        layers=[4, 10, 8, 3],  # 4 input, 10 hidden, 8 hidden, 3 output
        learning_rate=0.01,
        activation='relu',
        output_activation='softmax',
        max_iterations=500,
        batch_size=32,
        regularization='l2',
        lambda_reg=0.001
    )
    
    nn_classifier.fit(X_train, y_train)
    
    # Evaluate
    train_accuracy = nn_classifier.score(X_train, y_train)
    test_accuracy = nn_classifier.score(X_test, y_test)
    
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Plot training history
    nn_classifier.plot_training_history()
    
    # Visualize weights
    print("\n=== Weight Visualization ===")
    nn_classifier.visualize_weights(layer_idx=0)
    
    # 2D visualization
    print("\n=== 2D Visualization Example ===")
    X_2d, y_2d = make_circles(n_samples=500, noise=0.1, factor=0.3, random_state=42)
    
    # Convert to 3-class problem
    y_2d_3class = np.zeros_like(y_2d)
    y_2d_3class[y_2d == 0] = 0
    y_2d_3class[(y_2d == 1) & (X_2d[:, 0] > 0)] = 1
    y_2d_3class[(y_2d == 1) & (X_2d[:, 0] <= 0)] = 2
    
    nn_2d = NeuralNetwork(
        layers=[2, 8, 6, 3],
        learning_rate=0.1,
        activation='relu',
        output_activation='softmax',
        max_iterations=300,
        batch_size=16
    )
    
    nn_2d.fit(X_2d, y_2d_3class)
    
    # Plot decision boundary
    nn_2d.plot_decision_boundary(X_2d, y_2d_3class)
    
    print("\n=== Neural Network Regression Example ===")
    
    # Generate regression data
    np.random.seed(42)
    X_reg = np.random.randn(500, 2)
    y_reg = 2 * X_reg[:, 0] + 3 * X_reg[:, 1] + 0.5 * X_reg[:, 0] * X_reg[:, 1] + np.random.randn(500) * 0.1
    
    # Split data
    split_idx = int(0.8 * len(X_reg))
    X_train_reg, X_test_reg = X_reg[:split_idx], X_reg[split_idx:]
    y_train_reg, y_test_reg = y_reg[:split_idx], y_reg[split_idx:]
    
    # Create regression neural network
    nn_regressor = NeuralNetwork(
        layers=[2, 8, 4, 1],
        learning_rate=0.01,
        activation='tanh',
        output_activation='linear',
        max_iterations=400,
        batch_size=32,
        regularization='l2',
        lambda_reg=0.001
    )
    
    nn_regressor.fit(X_train_reg, y_train_reg)
    
    # Evaluate regression
    train_r2 = nn_regressor.score(X_train_reg, y_train_reg)
    test_r2 = nn_regressor.score(X_test_reg, y_test_reg)
    
    print(f"Train R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    
    # Plot regression results
    predictions_reg = nn_regressor.predict(X_test_reg)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test_reg, predictions_reg, alpha=0.6)
    plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Neural Network Regression: Predictions vs True Values')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    residuals = y_test_reg - predictions_reg
    plt.scatter(predictions_reg, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predictions')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Plot training history for regression
    nn_regressor.plot_training_history()
    
    print("\n=== Comparing Different Architectures ===")
    
    # Compare different network architectures
    architectures = [
        [2, 4, 3],
        [2, 8, 6, 3],
        [2, 16, 12, 8, 3],
        [2, 32, 16, 3]
    ]
    
    results = []
    
    for arch in architectures:
        print(f"Training architecture: {arch}")
        nn_comp = NeuralNetwork(
            layers=arch,
            learning_rate=0.05,
            activation='relu',
            output_activation='softmax',
            max_iterations=200,
            batch_size=32
        )
        
        nn_comp.fit(X_2d, y_2d_3class)
        accuracy = nn_comp.score(X_2d, y_2d_3class)
        results.append((arch, accuracy, len(nn_comp.cost_history)))
        
        print(f"Architecture {arch}: Accuracy = {accuracy:.4f}")
    
    # Plot architecture comparison
    arch_names = [str(arch) for arch, _, _ in results]
    accuracies = [acc for _, acc, _ in results]
    
    plt.figure(figsize=(12, 6))
    plt.bar(arch_names, accuracies, color=['skyblue', 'lightgreen', 'orange', 'pink'])
    plt.xlabel('Network Architecture')
    plt.ylabel('Accuracy')
    plt.title('Comparison of Different Network Architectures')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    for i, (arch, acc, _) in enumerate(results):
        plt.text(i, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== Activation Function Comparison ===")
    
    # Compare different activation functions
    activations = ['relu', 'sigmoid', 'tanh']
    activation_results = []
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, activation in enumerate(activations):
        print(f"Training with {activation} activation...")
        
        nn_act = NeuralNetwork(
            layers=[2, 10, 6, 3],
            learning_rate=0.05,
            activation=activation,
            output_activation='softmax',
            max_iterations=200,
            batch_size=32
        )
        
        nn_act.fit(X_2d, y_2d_3class)
        accuracy = nn_act.score(X_2d, y_2d_3class)
        activation_results.append((activation, accuracy))
        
        # Plot decision boundary for each activation
        plt.subplot(1, 3, i+1)
        
        colors = ['red', 'blue', 'green']
        for j, label in enumerate(np.unique(y_2d_3class)):
            mask = y_2d_3class == label
            plt.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                       c=colors[j], alpha=0.7, s=30, label=f'Class {j}')
        
        # Create mesh and plot decision boundary
        x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
        y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                            np.arange(y_min, y_max, 0.02))
        
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = nn_act.predict(mesh_points)
        Z = Z.reshape(xx.shape)
        
        plt.contourf(xx, yy, Z, alpha=0.3, levels=2)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(f'{activation.title()} Activation\nAccuracy: {accuracy:.3f}')
        plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    print("\nActivation Function Results:")
    for activation, accuracy in activation_results:
        print(f"{activation}: {accuracy:.4f}")
    
    print("\n=== Binary Classification Example ===")
    
    # Binary classification with sigmoid output
    X_binary, y_binary = make_moons(n_samples=400, noise=0.2, random_state=42)
    
    nn_binary = NeuralNetwork(
        layers=[2, 8, 4, 1],
        learning_rate=0.1,
        activation='relu',
        output_activation='sigmoid',
        max_iterations=300,
        batch_size=16
    )
    
    nn_binary.fit(X_binary, y_binary)
    
    binary_accuracy = nn_binary.score(X_binary, y_binary)
    print(f"Binary Classification Accuracy: {binary_accuracy:.4f}")
    
    # Plot binary classification results
    plt.figure(figsize=(12, 5))
    
    # Plot decision boundary
    plt.subplot(1, 2, 1)
    colors = ['red', 'blue']
    for i, label in enumerate(np.unique(y_binary)):
        mask = y_binary == label
        plt.scatter(X_binary[mask, 0], X_binary[mask, 1], 
                   c=colors[i], alpha=0.7, s=50, label=f'Class {label}')
    
    # Plot decision boundary
    x_min, x_max = X_binary[:, 0].min() - 1, X_binary[:, 0].max() + 1
    y_min, y_max = X_binary[:, 1].min() - 1, X_binary[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                        np.arange(y_min, y_max, 0.02))
    
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z_prob = nn_binary.predict_proba(mesh_points)
    Z_prob = Z_prob.reshape(xx.shape)
    
    # Plot probability contours
    contour = plt.contourf(xx, yy, Z_prob, levels=50, alpha=0.6, cmap='RdBu')
    plt.colorbar(contour, label='Probability')
    
    # Plot decision boundary
    plt.contour(xx, yy, Z_prob, levels=[0.5], colors=['black'], linewidths=2)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'Binary Neural Network Classification\nAccuracy: {binary_accuracy:.3f}')
    plt.legend()
    
    # Plot training history
    plt.subplot(1, 2, 2)
    plt.plot(nn_binary.cost_history, label='Cost')
    plt.plot(nn_binary.accuracy_history, label='Accuracy')
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Show some probability predictions
    print("\nSample probability predictions:")
    sample_probs = nn_binary.predict_proba(X_binary[:10])
    sample_preds = nn_binary.predict(X_binary[:10])
    
    for i in range(10):
        prob = sample_probs[i, 0] if len(sample_probs.shape) > 1 else sample_probs[i]
        print(f"Sample {i}: Prob(Class 1) = {prob:.3f}, "
              f"Predicted = {sample_preds[i]}, Actual = {y_binary[i]}")
    
    print("\n=== Regularization Comparison ===")
    
    # Compare different regularization techniques
    regularizations = [None, 'l1', 'l2']
    reg_results = []
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, reg in enumerate(regularizations):
        print(f"Training with {reg} regularization...")
        
        nn_reg = NeuralNetwork(
            layers=[2, 16, 12, 3],  # Larger network to show overfitting
            learning_rate=0.05,
            activation='relu',
            output_activation='softmax',
            max_iterations=300,
            batch_size=16,
            regularization=reg,
            lambda_reg=0.01
        )
        
        nn_reg.fit(X_2d, y_2d_3class)
        accuracy = nn_reg.score(X_2d, y_2d_3class)
        reg_results.append((reg, accuracy))
        
        # Plot training history
        plt.subplot(1, 3, i+1)
        plt.plot(nn_reg.cost_history, label='Cost')
        plt.plot(nn_reg.accuracy_history, label='Accuracy')
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.title(f'Regularization: {reg}\nFinal Accuracy: {accuracy:.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\nRegularization Results:")
    for reg, accuracy in reg_results:
        print(f"{reg if reg else 'None'}: {accuracy:.4f}")
    
    print("\n=== Analysis Complete ===")
    print("Neural Network implementation includes:")
    print("- Multi-layer architecture support")
    print("- Multiple activation functions (ReLU, Sigmoid, Tanh)")
    print("- Different output activations (Softmax, Sigmoid, Linear)")
    print("- Regularization techniques (L1, L2)")
    print("- Mini-batch gradient descent")
    print("- Comprehensive visualization tools")
    print("- Support for classification and regression")
    
    print("\n=== Network Architecture Analysis ===")
    
    # Demonstrate activation visualization
    print("Getting layer activations for sample data...")
    sample_data = X_2d[:5]  # First 5 samples
    
    # Get activations for all layers
    all_activations = nn_2d.get_activations(sample_data)
    
    print(f"Input shape: {all_activations[0].shape}")
    for i in range(1, len(all_activations)):
        print(f"Layer {i} activations shape: {all_activations[i].shape}")
        print(f"Layer {i} activation stats - Mean: {np.mean(all_activations[i]):.4f}, "
              f"Std: {np.std(all_activations[i]):.4f}")
    
    # Visualize activations for different layers
    plt.figure(figsize=(15, 10))
    
    for i in range(1, min(4, len(all_activations))):  # Show first 3 hidden layers
        plt.subplot(2, 3, i)
        
        # Plot activation heatmap
        activations = all_activations[i]
        plt.imshow(activations.T, cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.title(f'Layer {i} Activations\n({activations.shape[1]} neurons)')
        plt.xlabel('Sample Index')
        plt.ylabel('Neuron Index')
        
        # Plot activation distribution
        plt.subplot(2, 3, i + 3)
        plt.hist(activations.flatten(), bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Activation Value')
        plt.ylabel('Frequency')
        plt.title(f'Layer {i} Activation Distribution')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== Gradient Analysis ===")
    
    # Demonstrate gradient computation (simplified version)
    def analyze_gradients(network, X_sample, y_sample):
        """Analyze gradient magnitudes across layers"""
        # Forward pass
        activations, z_values = network._forward_propagation(X_sample)
        
        # Prepare labels for gradient computation
        if len(y_sample.shape) == 1 and network.output_activation == 'softmax':
            num_classes = len(np.unique(y_sample))
            y_sample = network._one_hot_encode(y_sample, num_classes)
        elif len(y_sample.shape) == 1 and network.output_activation == 'sigmoid':
            y_sample = y_sample.reshape(-1, 1)
        
        # Backward pass
        dW, db = network._backward_propagation(X_sample, y_sample, activations, z_values)
        
        # Calculate gradient magnitudes
        gradient_magnitudes = []
        for i, grad in enumerate(dW):
            grad_mag = np.mean(np.abs(grad))
            gradient_magnitudes.append(grad_mag)
            print(f"Layer {i} gradient magnitude: {grad_mag:.6f}")
        
        return gradient_magnitudes
    
    # Analyze gradients on a small sample
    sample_X = X_2d[:32]  # Small batch
    sample_y = y_2d_3class[:32]
    
    print("Analyzing gradient magnitudes...")
    grad_mags = analyze_gradients(nn_2d, sample_X, sample_y)
    
    # Plot gradient magnitudes
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(grad_mags)), grad_mags, alpha=0.7)
    plt.xlabel('Layer Index')
    plt.ylabel('Average Gradient Magnitude')
    plt.title('Gradient Magnitudes Across Layers')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("\n=== Learning Rate Sensitivity Analysis ===")
    
    # Test different learning rates
    learning_rates = [0.001, 0.01, 0.1, 0.5]
    lr_results = []
    
    plt.figure(figsize=(15, 10))
    
    for i, lr in enumerate(learning_rates):
        print(f"Testing learning rate: {lr}")
        
        nn_lr = NeuralNetwork(
            layers=[2, 8, 3],
            learning_rate=lr,
            activation='relu',
            output_activation='softmax',
            max_iterations=150,
            batch_size=16
        )
        
        nn_lr.fit(X_2d[:200], y_2d_3class[:200])  # Use subset for faster training
        final_accuracy = nn_lr.score(X_2d[:200], y_2d_3class[:200])
        lr_results.append((lr, final_accuracy))
        
        # Plot cost history
        plt.subplot(2, 2, i+1)
        plt.plot(nn_lr.cost_history, label='Cost')
        plt.plot(nn_lr.accuracy_history, label='Accuracy')
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.title(f'Learning Rate: {lr}\nFinal Accuracy: {final_accuracy:.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\nLearning Rate Results:")
    for lr, acc in lr_results:
        print(f"LR {lr}: {acc:.4f}")
    
    print("\n=== Batch Size Impact Analysis ===")
    
    # Test different batch sizes
    batch_sizes = [8, 16, 32, 64]
    batch_results = []
    
    plt.figure(figsize=(15, 10))
    
    for i, batch_size in enumerate(batch_sizes):
        print(f"Testing batch size: {batch_size}")
        
        nn_batch = NeuralNetwork(
            layers=[2, 8, 3],
            learning_rate=0.05,
            activation='relu',
            output_activation='softmax',
            max_iterations=150,
            batch_size=batch_size
        )
        
        nn_batch.fit(X_2d[:300], y_2d_3class[:300])
        final_accuracy = nn_batch.score(X_2d[:300], y_2d_3class[:300])
        batch_results.append((batch_size, final_accuracy))
        
        # Plot training history
        plt.subplot(2, 2, i+1)
        plt.plot(nn_batch.cost_history, label='Cost')
        plt.plot(nn_batch.accuracy_history, label='Accuracy')
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.title(f'Batch Size: {batch_size}\nFinal Accuracy: {final_accuracy:.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\nBatch Size Results:")
    for bs, acc in batch_results:
        print(f"Batch Size {bs}: {acc:.4f}")
    
    print("\n=== Network Depth Analysis ===")
    
    # Test different network depths
    depths = [
        [2, 3],           # 1 hidden layer
        [2, 8, 3],        # 2 hidden layers  
        [2, 8, 6, 3],     # 3 hidden layers
        [2, 8, 6, 4, 3]   # 4 hidden layers
    ]
    
    depth_results = []
    
    plt.figure(figsize=(15, 10))
    
    for i, architecture in enumerate(depths):
        print(f"Testing architecture: {architecture}")
        
        nn_depth = NeuralNetwork(
            layers=architecture,
            learning_rate=0.05,
            activation='relu',
            output_activation='softmax',
            max_iterations=200,
            batch_size=32
        )
        
        nn_depth.fit(X_2d, y_2d_3class)
        final_accuracy = nn_depth.score(X_2d, y_2d_3class)
        depth_results.append((len(architecture)-2, final_accuracy))  # -2 for input and output
        
        # Plot training curves
        plt.subplot(2, 2, i+1)
        plt.plot(nn_depth.cost_history, label='Cost')
        plt.plot(nn_depth.accuracy_history, label='Accuracy')
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.title(f'Depth: {len(architecture)-2} hidden layers\nAccuracy: {final_accuracy:.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\nNetwork Depth Results:")
    for depth, acc in depth_results:
        print(f"{depth} hidden layers: {acc:.4f}")
    
    print("\n=== Overfitting Analysis ===")
    
    # Demonstrate overfitting with train/validation split
    # Use smaller dataset to encourage overfitting
    X_small = X_2d[:150]
    y_small = y_2d_3class[:150]
    
    # Split into train/validation
    val_split = int(0.7 * len(X_small))
    X_train_small = X_small[:val_split]
    y_train_small = y_small[:val_split]
    X_val_small = X_small[val_split:]
    y_val_small = y_small[val_split:]
    
    # Train a large network (prone to overfitting)
    nn_overfit = NeuralNetwork(
        layers=[2, 32, 16, 8, 3],  # Large network
        learning_rate=0.1,
        activation='relu',
        output_activation='softmax',
        max_iterations=500,
        batch_size=8,  # Small batch size
        regularization=None  # No regularization
    )
    
    # Manual training loop to track validation performance
    train_accuracies = []
    val_accuracies = []
    
    print("Training large network to demonstrate overfitting...")
    
    # Fit the network
    nn_overfit.fit(X_train_small, y_train_small)
    
    # Compute accuracies at each iteration (simplified)
    for i in range(0, len(nn_overfit.cost_history), 10):
        # Use current weights to compute validation accuracy
        val_acc = nn_overfit.score(X_val_small, y_val_small)
        train_acc = nn_overfit.score(X_train_small, y_train_small)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
    
    # Plot overfitting curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(nn_overfit.cost_history)
    plt.xlabel('Iteration')
    plt.ylabel('Training Cost')
    plt.title('Training Cost (Overfitting Demo)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    iterations = range(0, len(nn_overfit.cost_history), 10)[:len(train_accuracies)]
    plt.plot(iterations, train_accuracies, label='Training Accuracy')
    plt.plot(iterations, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    # Show decision boundary of overfitted model
    colors = ['red', 'blue', 'green']
    for j, label in enumerate(np.unique(y_small)):
        mask = y_small == label
        plt.scatter(X_small[mask, 0], X_small[mask, 1], 
                   c=colors[j], alpha=0.7, s=30, label=f'Class {j}')
    
    # Create mesh and plot decision boundary
    x_min, x_max = X_small[:, 0].min() - 0.5, X_small[:, 0].max() + 0.5
    y_min, y_max = X_small[:, 1].min() - 0.5, X_small[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                        np.arange(y_min, y_max, 0.02))
    
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = nn_overfit.predict(mesh_points)
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, levels=2)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Overfitted Decision Boundary')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    final_train_acc = nn_overfit.score(X_train_small, y_train_small)
    final_val_acc = nn_overfit.score(X_val_small, y_val_small)
    
    print(f"Final Training Accuracy: {final_train_acc:.4f}")
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")
    print(f"Overfitting Gap: {final_train_acc - final_val_acc:.4f}")
