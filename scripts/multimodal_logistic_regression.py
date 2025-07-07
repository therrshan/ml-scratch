import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union

class MultimodalLogisticRegression:
    """
    Multimodal Logistic Regression implementation with various fusion strategies
    Supports early, late, and intermediate fusion of multiple data modalities
    """
    
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6, 
                 fusion_type='early', modality_weights=None, hidden_dim=None):
        """
        Initialize Multimodal Logistic Regression model
        
        Parameters:
        learning_rate (float): Step size for gradient descent
        max_iterations (int): Maximum number of iterations
        tolerance (float): Convergence tolerance
        fusion_type (str): Type of fusion - 'early', 'late', or 'intermediate'
        modality_weights (list): Weights for late fusion (must sum to 1)
        hidden_dim (int): Hidden dimension for intermediate fusion
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.fusion_type = fusion_type
        self.modality_weights = modality_weights
        self.hidden_dim = hidden_dim
        
        # Model parameters
        self.modality_models = {}
        self.fusion_weights = None
        self.fusion_bias = None
        self.cost_history = []
        
    def _add_bias(self, X):
        """Add bias column to feature matrix"""
        return np.c_[np.ones(X.shape[0]), X]
    
    def _sigmoid(self, z):
        """Sigmoid activation function"""
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))
    
    def _relu(self, z):
        """ReLU activation function"""
        return np.maximum(0, z)
    
    def _compute_cost(self, predictions, y):
        """Compute logistic regression cost (cross-entropy)"""
        m = len(y)
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        cost = -(1/m) * np.sum(y * np.log(predictions) + (1-y) * np.log(1-predictions))
        return cost
    
    def _initialize_modality_models(self, modalities):
        """Initialize separate models for each modality"""
        for name, X in modalities.items():
            n_features = X.shape[1] + 1  # +1 for bias
            self.modality_models[name] = {
                'theta': np.random.normal(0, 0.01, n_features),
                'input_dim': X.shape[1]
            }
    
    def _early_fusion(self, modalities):
        """Early fusion: concatenate all modalities before processing"""
        # Concatenate all modalities
        fused_features = np.hstack([modalities[name] for name in sorted(modalities.keys())])
        return fused_features
    
    def _late_fusion_forward(self, modalities):
        """Late fusion: process each modality separately then combine predictions"""
        predictions = []
        
        for name in sorted(modalities.keys()):
            X = modalities[name]
            X_with_bias = self._add_bias(X)
            z = X_with_bias.dot(self.modality_models[name]['theta'])
            pred = self._sigmoid(z)
            predictions.append(pred)
        
        # Weighted average of predictions
        if self.modality_weights is None:
            # Equal weights if not specified
            weights = np.ones(len(predictions)) / len(predictions)
        else:
            weights = np.array(self.modality_weights)
        
        final_prediction = np.sum([w * p for w, p in zip(weights, predictions)], axis=0)
        return final_prediction, predictions
    
    def _intermediate_fusion_forward(self, modalities):
        """Intermediate fusion: extract features from each modality then fuse"""
        intermediate_features = []
        
        # Extract intermediate features from each modality
        for name in sorted(modalities.keys()):
            X = modalities[name]
            X_with_bias = self._add_bias(X)
            
            # Get intermediate representation (before sigmoid)
            z = X_with_bias.dot(self.modality_models[name]['theta'])
            # For intermediate fusion, we'll use the raw scores as features
            intermediate_features.append(z.reshape(-1, 1))  # Make it a column vector
        
        # Concatenate intermediate features
        fused_features = np.hstack(intermediate_features)
        
        # Apply fusion layer
        z_fusion = fused_features.dot(self.fusion_weights) + self.fusion_bias
        final_prediction = self._sigmoid(z_fusion)
        
        return final_prediction, intermediate_features
    
    def fit(self, modalities: Dict[str, np.ndarray], y: np.ndarray):
        """
        Train the multimodal logistic regression model
        
        Parameters:
        modalities (dict): Dictionary of modality names to feature arrays
                          e.g., {'visual': X_visual, 'text': X_text}
        y (array): Training targets (m,) - binary (0 or 1)
        """
        # Convert to numpy array
        y = np.array(y)
        m = y.shape[0]
        
        # Initialize models based on fusion type
        if self.fusion_type == 'early':
            # Early fusion: single model for concatenated features
            fused_X = self._early_fusion(modalities)
            n_features = fused_X.shape[1] + 1  # +1 for bias
            self.fusion_weights = np.random.normal(0, 0.01, n_features)
            
        elif self.fusion_type == 'late':
            # Late fusion: separate models for each modality
            self._initialize_modality_models(modalities)
            
        elif self.fusion_type == 'intermediate':
            # Intermediate fusion: separate feature extractors + fusion layer
            self._initialize_modality_models(modalities)
            
            # Calculate fusion layer dimensions
            # The fusion input dimension is the number of modalities (one score per modality)
            fusion_input_dim = len(modalities)
            
            self.fusion_weights = np.random.normal(0, 0.01, fusion_input_dim)
            self.fusion_bias = 0.0
        
        # Training loop
        prev_cost = float('inf')
        
        for iteration in range(self.max_iterations):
            # Forward pass based on fusion type
            if self.fusion_type == 'early':
                # Early fusion forward pass
                fused_X = self._early_fusion(modalities)
                X_with_bias = self._add_bias(fused_X)
                z = X_with_bias.dot(self.fusion_weights)
                predictions = self._sigmoid(z)
                
                # Compute gradients
                gradients = (1/m) * X_with_bias.T.dot(predictions - y)
                self.fusion_weights -= self.learning_rate * gradients
                
            elif self.fusion_type == 'late':
                # Late fusion forward pass
                predictions, modality_predictions = self._late_fusion_forward(modalities)
                
                # Update each modality model
                for i, name in enumerate(sorted(modalities.keys())):
                    X = modalities[name]
                    X_with_bias = self._add_bias(X)
                    
                    # Gradient for this modality
                    if self.modality_weights is None:
                        weight = 1.0 / len(modalities)
                    else:
                        weight = self.modality_weights[i]
                    
                    grad = (weight/m) * X_with_bias.T.dot(modality_predictions[i] - y)
                    self.modality_models[name]['theta'] -= self.learning_rate * grad
                
            elif self.fusion_type == 'intermediate':
                # Intermediate fusion forward pass
                predictions, intermediate_features = self._intermediate_fusion_forward(modalities)
                
                # Backpropagation through fusion layer
                fusion_grad = (1/m) * (predictions - y)
                
                # Update fusion layer
                fused_features = np.hstack(intermediate_features)
                self.fusion_weights -= self.learning_rate * fused_features.T.dot(fusion_grad)
                self.fusion_bias -= self.learning_rate * np.sum(fusion_grad)
                
                # Update modality-specific layers
                for i, name in enumerate(sorted(modalities.keys())):
                    X = modalities[name]
                    X_with_bias = self._add_bias(X)
                    
                    # Gradient through fusion weights
                    modality_grad = fusion_grad * self.fusion_weights[i]
                    
                    # Update modality parameters
                    grad = (1/m) * X_with_bias.T.dot(modality_grad)
                    self.modality_models[name]['theta'] -= self.learning_rate * grad
            
            # Compute and store cost
            cost = self._compute_cost(predictions, y)
            self.cost_history.append(cost)
            
            # Check convergence
            if abs(prev_cost - cost) < self.tolerance:
                print(f"Converged after {iteration+1} iterations")
                break
            prev_cost = cost
    
    def predict_proba(self, modalities: Dict[str, np.ndarray]):
        """
        Predict class probabilities
        
        Parameters:
        modalities (dict): Dictionary of modality names to feature arrays
        
        Returns:
        array: Predicted probabilities
        """
        if self.fusion_type == 'early':
            fused_X = self._early_fusion(modalities)
            X_with_bias = self._add_bias(fused_X)
            z = X_with_bias.dot(self.fusion_weights)
            return self._sigmoid(z)
        
        elif self.fusion_type == 'late':
            predictions, _ = self._late_fusion_forward(modalities)
            return predictions
        
        elif self.fusion_type == 'intermediate':
            predictions, _ = self._intermediate_fusion_forward(modalities)
            return predictions
    
    def predict(self, modalities: Dict[str, np.ndarray], threshold=0.5):
        """
        Make binary predictions
        
        Parameters:
        modalities (dict): Dictionary of modality names to feature arrays
        threshold (float): Decision threshold
        
        Returns:
        array: Binary predictions (0 or 1)
        """
        probabilities = self.predict_proba(modalities)
        return (probabilities >= threshold).astype(int)
    
    def score(self, modalities: Dict[str, np.ndarray], y: np.ndarray):
        """
        Calculate accuracy score
        
        Parameters:
        modalities (dict): Dictionary of modality names to feature arrays
        y (array): True labels
        
        Returns:
        float: Accuracy score
        """
        predictions = self.predict(modalities)
        return np.mean(predictions == y)
    
    def get_modality_importance(self):
        """Get importance scores for each modality (only for late fusion)"""
        if self.fusion_type != 'late':
            print("Modality importance only available for late fusion")
            return None
        
        if self.modality_weights is None:
            return {name: 1.0/len(self.modality_models) for name in self.modality_models}
        else:
            return {name: weight for name, weight in 
                   zip(sorted(self.modality_models.keys()), self.modality_weights)}
    
    def plot_cost_history(self):
        """Plot the cost function over iterations"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history)
        plt.title(f'Cost Function Over Iterations ({self.fusion_type} fusion)')
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.grid(True)
        plt.show()


# Example usage and demonstration
if __name__ == "__main__":
    # Generate synthetic multimodal data
    np.random.seed(42)
    n_samples = 1000
    
    # Modality 1: Visual features (e.g., image features)
    visual_dim = 50
    X_visual = np.random.randn(n_samples, visual_dim)
    
    # Modality 2: Text features (e.g., word embeddings)
    text_dim = 30
    X_text = np.random.randn(n_samples, text_dim)
    
    # Modality 3: Audio features (e.g., spectral features)
    audio_dim = 20
    X_audio = np.random.randn(n_samples, audio_dim)
    
    # Generate labels based on a combination of modalities
    # True relationship: visual features have highest importance
    y = (0.5 * X_visual[:, 0] + 0.3 * X_text[:, 0] + 0.2 * X_audio[:, 0] + 
         np.random.normal(0, 0.1, n_samples)) > 0
    y = y.astype(int)
    
    # Prepare modalities dictionary
    modalities = {
        'visual': X_visual,
        'text': X_text,
        'audio': X_audio
    }
    
    # Split data
    split_idx = int(0.8 * n_samples)
    train_modalities = {name: X[:split_idx] for name, X in modalities.items()}
    test_modalities = {name: X[split_idx:] for name, X in modalities.items()}
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Test different fusion strategies
    fusion_types = ['early', 'late', 'intermediate']
    results = {}
    
    for fusion_type in fusion_types:
        print(f"\n--- Testing {fusion_type.upper()} FUSION ---")
        
        # Create model
        if fusion_type == 'late':
            # Use custom weights for late fusion
            model = MultimodalLogisticRegression(
                learning_rate=0.1, 
                max_iterations=1000,
                fusion_type=fusion_type,
                modality_weights=[0.5, 0.3, 0.2]  # Matching true importance
            )
        else:
            model = MultimodalLogisticRegression(
                learning_rate=0.1, 
                max_iterations=1000,
                fusion_type=fusion_type,
                hidden_dim=10
            )
        
        # Train model
        model.fit(train_modalities, y_train)
        
        # Evaluate
        train_acc = model.score(train_modalities, y_train)
        test_acc = model.score(test_modalities, y_test)
        
        results[fusion_type] = {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'model': model
        }
        
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        
        if fusion_type == 'late':
            importance = model.get_modality_importance()
            print(f"Modality Importance: {importance}")
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Cost histories
    plt.subplot(1, 3, 1)
    for fusion_type in fusion_types:
        plt.plot(results[fusion_type]['model'].cost_history, 
                label=f'{fusion_type.capitalize()} fusion')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Training Cost Comparison')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Accuracy comparison
    plt.subplot(1, 3, 2)
    fusion_names = [f.capitalize() for f in fusion_types]
    train_accs = [results[f]['train_acc'] for f in fusion_types]
    test_accs = [results[f]['test_acc'] for f in fusion_types]
    
    x = np.arange(len(fusion_types))
    width = 0.35
    
    plt.bar(x - width/2, train_accs, width, label='Train', alpha=0.8)
    plt.bar(x + width/2, test_accs, width, label='Test', alpha=0.8)
    plt.xlabel('Fusion Type')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison')
    plt.xticks(x, fusion_names)
    plt.legend()
    plt.grid(True, axis='y')
    
    # Plot 3: Feature importance visualization (for late fusion)
    plt.subplot(1, 3, 3)
    late_model = results['late']['model']
    importance = late_model.get_modality_importance()
    if importance:
        modality_names = list(importance.keys())
        importance_values = list(importance.values())
        
        plt.bar(modality_names, importance_values, alpha=0.8)
        plt.xlabel('Modality')
        plt.ylabel('Importance Weight')
        plt.title('Modality Importance (Late Fusion)')
        plt.ylim(0, 0.6)
        plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    # Demonstrate prediction with missing modality
    print("\n--- Testing with Missing Modality ---")
    # Remove audio modality
    partial_test = {
        'visual': test_modalities['visual'],
        'text': test_modalities['text'],
        'audio': np.zeros_like(test_modalities['audio'])  # Zero out audio
    }
    
    for fusion_type in fusion_types:
        partial_acc = results[fusion_type]['model'].score(partial_test, y_test)
        print(f"{fusion_type.capitalize()} fusion accuracy (no audio): {partial_acc:.4f}")