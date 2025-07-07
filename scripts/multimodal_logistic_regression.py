"""
Multimodal logistic regression with early, late, and intermediate fusion using gradient descent.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

class MultimodalLogisticRegression:
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6, 
                 fusion_type='early', modality_weights=None, hidden_dim=None):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.fusion_type = fusion_type
        self.modality_weights = modality_weights
        self.hidden_dim = hidden_dim
        self.modality_models = {}
        self.fusion_weights = None
        self.fusion_bias = None
        self.cost_history = []

    def _add_bias(self, X):
        return np.c_[np.ones(X.shape[0]), X]

    def _sigmoid(self, z):
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))

    def _relu(self, z):
        return np.maximum(0, z)

    def _compute_cost(self, predictions, y):
        m = len(y)
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        return -(1/m) * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))

    def _initialize_modality_models(self, modalities):
        for name, X in modalities.items():
            n_features = X.shape[1] + 1
            self.modality_models[name] = {
                'theta': np.random.normal(0, 0.01, n_features),
                'input_dim': X.shape[1]
            }

    def _early_fusion(self, modalities):
        return np.hstack([modalities[name] for name in sorted(modalities.keys())])

    def _late_fusion_forward(self, modalities):
        predictions = []
        for name in sorted(modalities.keys()):
            X = modalities[name]
            X_with_bias = self._add_bias(X)
            z = X_with_bias.dot(self.modality_models[name]['theta'])
            pred = self._sigmoid(z)
            predictions.append(pred)
        weights = np.ones(len(predictions)) / len(predictions) if self.modality_weights is None else np.array(self.modality_weights)
        final_prediction = np.sum([w * p for w, p in zip(weights, predictions)], axis=0)
        return final_prediction, predictions

    def _intermediate_fusion_forward(self, modalities):
        intermediate_features = []
        for name in sorted(modalities.keys()):
            X = modalities[name]
            X_with_bias = self._add_bias(X)
            z = X_with_bias.dot(self.modality_models[name]['theta'])
            intermediate_features.append(z.reshape(-1, 1))
        fused_features = np.hstack(intermediate_features)
        z_fusion = fused_features.dot(self.fusion_weights) + self.fusion_bias
        final_prediction = self._sigmoid(z_fusion)
        return final_prediction, intermediate_features

    def fit(self, modalities: Dict[str, np.ndarray], y: np.ndarray):
        y = np.array(y)
        m = y.shape[0]

        if self.fusion_type == 'early':
            fused_X = self._early_fusion(modalities)
            n_features = fused_X.shape[1] + 1
            self.fusion_weights = np.random.normal(0, 0.01, n_features)
        elif self.fusion_type == 'late':
            self._initialize_modality_models(modalities)
        elif self.fusion_type == 'intermediate':
            self._initialize_modality_models(modalities)
            fusion_input_dim = len(modalities)
            self.fusion_weights = np.random.normal(0, 0.01, fusion_input_dim)
            self.fusion_bias = 0.0

        prev_cost = float('inf')
        for iteration in range(self.max_iterations):
            if self.fusion_type == 'early':
                fused_X = self._early_fusion(modalities)
                X_with_bias = self._add_bias(fused_X)
                z = X_with_bias.dot(self.fusion_weights)
                predictions = self._sigmoid(z)
                gradients = (1/m) * X_with_bias.T.dot(predictions - y)
                self.fusion_weights -= self.learning_rate * gradients
            elif self.fusion_type == 'late':
                predictions, modality_predictions = self._late_fusion_forward(modalities)
                for i, name in enumerate(sorted(modalities.keys())):
                    X = modalities[name]
                    X_with_bias = self._add_bias(X)
                    weight = 1.0 / len(modalities) if self.modality_weights is None else self.modality_weights[i]
                    grad = (weight/m) * X_with_bias.T.dot(modality_predictions[i] - y)
                    self.modality_models[name]['theta'] -= self.learning_rate * grad
            elif self.fusion_type == 'intermediate':
                predictions, intermediate_features = self._intermediate_fusion_forward(modalities)
                fusion_grad = (1/m) * (predictions - y)
                fused_features = np.hstack(intermediate_features)
                self.fusion_weights -= self.learning_rate * fused_features.T.dot(fusion_grad)
                self.fusion_bias -= self.learning_rate * np.sum(fusion_grad)
                for i, name in enumerate(sorted(modalities.keys())):
                    X = modalities[name]
                    X_with_bias = self._add_bias(X)
                    modality_grad = fusion_grad * self.fusion_weights[i]
                    grad = (1/m) * X_with_bias.T.dot(modality_grad)
                    self.modality_models[name]['theta'] -= self.learning_rate * grad

            cost = self._compute_cost(predictions, y)
            self.cost_history.append(cost)
            if abs(prev_cost - cost) < self.tolerance:
                print(f"Converged after {iteration+1} iterations")
                break
            prev_cost = cost

    def predict_proba(self, modalities: Dict[str, np.ndarray]):
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
        return (self.predict_proba(modalities) >= threshold).astype(int)

    def score(self, modalities: Dict[str, np.ndarray], y: np.ndarray):
        predictions = self.predict(modalities)
        return np.mean(predictions == y)

    def get_modality_importance(self):
        if self.fusion_type != 'late':
            print("Modality importance only available for late fusion")
            return None
        if self.modality_weights is None:
            return {name: 1.0/len(self.modality_models) for name in self.modality_models}
        else:
            return {name: weight for name, weight in zip(sorted(self.modality_models.keys()), self.modality_weights)}

    def plot_cost_history(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history)
        plt.title(f'Cost Function Over Iterations ({self.fusion_type} fusion)')
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    np.random.seed(42)
    n_samples = 1000
    visual_dim = 50
    X_visual = np.random.randn(n_samples, visual_dim)
    text_dim = 30
    X_text = np.random.randn(n_samples, text_dim)
    audio_dim = 20
    X_audio = np.random.randn(n_samples, audio_dim)
    y = (0.5 * X_visual[:, 0] + 0.3 * X_text[:, 0] + 0.2 * X_audio[:, 0] + np.random.normal(0, 0.1, n_samples)) > 0
    y = y.astype(int)
    modalities = {'visual': X_visual, 'text': X_text, 'audio': X_audio}
    split_idx = int(0.8 * n_samples)
    train_modalities = {name: X[:split_idx] for name, X in modalities.items()}
    test_modalities = {name: X[split_idx:] for name, X in modalities.items()}
    y_train, y_test = y[:split_idx], y[split_idx:]
    fusion_types = ['early', 'late', 'intermediate']
    results = {}
    for fusion_type in fusion_types:
        print(f"\n--- Testing {fusion_type.upper()} FUSION ---")
        if fusion_type == 'late':
            model = MultimodalLogisticRegression(
                learning_rate=0.1, max_iterations=1000,
                fusion_type=fusion_type, modality_weights=[0.5, 0.3, 0.2])
        else:
            model = MultimodalLogisticRegression(
                learning_rate=0.1, max_iterations=1000,
                fusion_type=fusion_type, hidden_dim=10)
        model.fit(train_modalities, y_train)
        train_acc = model.score(train_modalities, y_train)
        test_acc = model.score(test_modalities, y_test)
        results[fusion_type] = {'train_acc': train_acc, 'test_acc': test_acc, 'model': model}
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        if fusion_type == 'late':
            importance = model.get_modality_importance()
            print(f"Modality Importance: {importance}")
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    for fusion_type in fusion_types:
        plt.plot(results[fusion_type]['model'].cost_history, label=f'{fusion_type.capitalize()} fusion')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Training Cost Comparison')
    plt.legend()
    plt.grid(True)
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
    print("\n--- Testing with Missing Modality ---")
    partial_test = {
        'visual': test_modalities['visual'],
        'text': test_modalities['text'],
        'audio': np.zeros_like(test_modalities['audio'])
    }
    for fusion_type in fusion_types:
        partial_acc = results[fusion_type]['model'].score(partial_test, y_test)
        print(f"{fusion_type.capitalize()} fusion accuracy (no audio): {partial_acc:.4f}")
