import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes Classifier implementation from scratch
    Assumes features follow a Gaussian distribution
    """
    
    def __init__(self):
        """Initialize Gaussian Naive Bayes classifier"""
        self.classes = None
        self.class_priors = {}
        self.feature_stats = {}  # {class: {feature: {'mean': mean, 'var': var}}}
        
    def fit(self, X, y):
        """
        Train the Naive Bayes classifier
        
        Parameters:
        X (array): Training features (m x n)
        y (array): Training labels (m,)
        """
        X = np.array(X)
        y = np.array(y)
        
        self.classes = np.unique(y)
        n_samples = len(y)
        n_features = X.shape[1]
        
        # Calculate class priors P(class)
        for cls in self.classes:
            self.class_priors[cls] = np.sum(y == cls) / n_samples
        
        # Calculate feature statistics for each class
        self.feature_stats = {}
        for cls in self.classes:
            self.feature_stats[cls] = {}
            class_samples = X[y == cls]
            
            for feature_idx in range(n_features):
                feature_values = class_samples[:, feature_idx]
                self.feature_stats[cls][feature_idx] = {
                    'mean': np.mean(feature_values),
                    'var': np.var(feature_values) + 1e-9  # Add small epsilon to avoid division by zero
                }
        
        print(f"Gaussian Naive Bayes fitted on {n_samples} samples with {len(self.classes)} classes")
    
    def _gaussian_probability(self, x, mean, var):
        """Calculate Gaussian probability density"""
        coefficient = 1 / np.sqrt(2 * np.pi * var)
        exponent = np.exp(-((x - mean) ** 2) / (2 * var))
        return coefficient * exponent
    
    def _predict_single(self, x):
        """Predict class probabilities for a single sample"""
        posteriors = {}
        
        for cls in self.classes:
            # Start with class prior
            posterior = np.log(self.class_priors[cls])
            
            # Multiply by feature likelihoods (add in log space)
            for feature_idx, feature_value in enumerate(x):
                mean = self.feature_stats[cls][feature_idx]['mean']
                var = self.feature_stats[cls][feature_idx]['var']
                likelihood = self._gaussian_probability(feature_value, mean, var)
                posterior += np.log(likelihood + 1e-10)  # Add small epsilon
            
            posteriors[cls] = posterior
        
        return posteriors
    
    def predict_proba(self, X):
        """
        Predict class probabilities
        
        Parameters:
        X (array): Features to predict on
        
        Returns:
        array: Class probabilities
        """
        X = np.array(X)
        probabilities = []
        
        for x in X:
            posteriors = self._predict_single(x)
            
            # Convert log probabilities to actual probabilities
            max_log_prob = max(posteriors.values())
            normalized_probs = {}
            
            for cls in self.classes:
                normalized_probs[cls] = np.exp(posteriors[cls] - max_log_prob)
            
            # Normalize to sum to 1
            total_prob = sum(normalized_probs.values())
            probs = [normalized_probs[cls] / total_prob for cls in self.classes]
            probabilities.append(probs)
        
        return np.array(probabilities)
    
    def predict(self, X):
        """
        Make class predictions
        
        Parameters:
        X (array): Features to predict on
        
        Returns:
        array: Predicted class labels
        """
        X = np.array(X)
        predictions = []
        
        for x in X:
            posteriors = self._predict_single(x)
            predicted_class = max(posteriors, key=posteriors.get)
            predictions.append(predicted_class)
        
        return np.array(predictions)
    
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
    
    def plot_feature_distributions(self, X, y, feature_names=None):
        """
        Plot feature distributions for each class
        
        Parameters:
        X (array): Features
        y (array): Labels
        feature_names (list): Names of features
        """
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(X.shape[1])]
        
        n_features = X.shape[1]
        n_classes = len(self.classes)
        
        fig, axes = plt.subplots(n_features, 1, figsize=(12, 4 * n_features))
        if n_features == 1:
            axes = [axes]
        
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        
        for feature_idx in range(n_features):
            ax = axes[feature_idx]
            
            for i, cls in enumerate(self.classes):
                class_data = X[y == cls, feature_idx]
                
                # Plot histogram
                ax.hist(class_data, alpha=0.6, bins=20, 
                       color=colors[i % len(colors)], label=f'Class {cls}')
                
                # Plot Gaussian curve
                mean = self.feature_stats[cls][feature_idx]['mean']
                var = self.feature_stats[cls][feature_idx]['var']
                std = np.sqrt(var)
                
                x_range = np.linspace(class_data.min() - 2*std, 
                                    class_data.max() + 2*std, 100)
                gaussian_curve = [self._gaussian_probability(x, mean, var) for x in x_range]
                
                # Scale the curve to match histogram
                scale_factor = len(class_data) * (class_data.max() - class_data.min()) / 20
                gaussian_curve = np.array(gaussian_curve) * scale_factor
                
                ax.plot(x_range, gaussian_curve, color=colors[i % len(colors)], 
                       linewidth=2, linestyle='--')
            
            ax.set_xlabel(feature_names[feature_idx])
            ax.set_ylabel('Frequency / Density')
            ax.set_title(f'Distribution of {feature_names[feature_idx]}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_decision_boundary(self, X, y, resolution=0.02):
        """Plot decision boundary for 2D data"""
        if X.shape[1] != 2:
            print("Decision boundary plot only available for 2D data")
            return
        
        # Create color map
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        unique_labels = np.unique(y)
        color_map = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}
        
        plt.figure(figsize=(12, 8))
        
        # Plot data points
        for label in unique_labels:
            mask = y == label
            plt.scatter(X[mask, 0], X[mask, 1], 
                       c=color_map[label], label=f'Class {label}', alpha=0.7)
        
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
        plt.title('Gaussian Naive Bayes Decision Boundary')
        plt.legend()
        plt.colorbar()
        plt.show()


class MultinomialNaiveBayes:
    """
    Multinomial Naive Bayes Classifier implementation from scratch
    Suitable for discrete features (e.g., word counts)
    """
    
    def __init__(self, alpha=1.0):
        """
        Initialize Multinomial Naive Bayes classifier
        
        Parameters:
        alpha (float): Laplace smoothing parameter
        """
        self.alpha = alpha
        self.classes = None
        self.class_priors = {}
        self.feature_probs = {}  # {class: {feature: probability}}
        
    def fit(self, X, y):
        """
        Train the Multinomial Naive Bayes classifier
        
        Parameters:
        X (array): Training features (m x n) - should be non-negative
        y (array): Training labels (m,)
        """
        X = np.array(X)
        y = np.array(y)
        
        self.classes = np.unique(y)
        n_samples = len(y)
        n_features = X.shape[1]
        
        # Calculate class priors P(class)
        for cls in self.classes:
            self.class_priors[cls] = np.sum(y == cls) / n_samples
        
        # Calculate feature probabilities for each class with Laplace smoothing
        self.feature_probs = {}
        for cls in self.classes:
            self.feature_probs[cls] = {}
            class_samples = X[y == cls]
            
            # Sum of all feature values for this class
            class_feature_sums = np.sum(class_samples, axis=0)
            # Total count for this class (sum of all features for all samples)
            total_count = np.sum(class_feature_sums)
            
            # Calculate probabilities with Laplace smoothing
            for feature_idx in range(n_features):
                numerator = class_feature_sums[feature_idx] + self.alpha
                denominator = total_count + self.alpha * n_features
                self.feature_probs[cls][feature_idx] = numerator / denominator
        
        print(f"Multinomial Naive Bayes fitted on {n_samples} samples with {len(self.classes)} classes")
    
    def _predict_single(self, x):
        """Predict class probabilities for a single sample"""
        posteriors = {}
        
        for cls in self.classes:
            # Start with class prior (in log space)
            posterior = np.log(self.class_priors[cls])
            
            # Add feature log-likelihoods
            for feature_idx, feature_count in enumerate(x):
                if feature_count > 0:  # Only consider non-zero features
                    feature_prob = self.feature_probs[cls][feature_idx]
                    posterior += feature_count * np.log(feature_prob)
            
            posteriors[cls] = posterior
        
        return posteriors
    
    def predict_proba(self, X):
        """
        Predict class probabilities
        
        Parameters:
        X (array): Features to predict on
        
        Returns:
        array: Class probabilities
        """
        X = np.array(X)
        probabilities = []
        
        for x in X:
            posteriors = self._predict_single(x)
            
            # Convert log probabilities to actual probabilities
            max_log_prob = max(posteriors.values())
            normalized_probs = {}
            
            for cls in self.classes:
                normalized_probs[cls] = np.exp(posteriors[cls] - max_log_prob)
            
            # Normalize to sum to 1
            total_prob = sum(normalized_probs.values())
            probs = [normalized_probs[cls] / total_prob for cls in self.classes]
            probabilities.append(probs)
        
        return np.array(probabilities)
    
    def predict(self, X):
        """
        Make class predictions
        
        Parameters:
        X (array): Features to predict on
        
        Returns:
        array: Predicted class labels
        """
        X = np.array(X)
        predictions = []
        
        for x in X:
            posteriors = self._predict_single(x)
            predicted_class = max(posteriors, key=posteriors.get)
            predictions.append(predicted_class)
        
        return np.array(predictions)
    
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

# Example usage
if __name__ == "__main__":
    print("=== Gaussian Naive Bayes Example ===")
    
    # Generate continuous data for Gaussian NB
    np.random.seed(42)
    from sklearn.datasets import make_classification
    X_gaussian, y_gaussian = make_classification(n_samples=500, n_features=4, 
                                               n_redundant=0, n_informative=4,
                                               n_clusters_per_class=1, n_classes=3, 
                                               random_state=42)
    
    # Split data
    split_idx = int(0.8 * len(X_gaussian))
    X_train, X_test = X_gaussian[:split_idx], X_gaussian[split_idx:]
    y_train, y_test = y_gaussian[:split_idx], y_gaussian[split_idx:]
    
    # Create and train Gaussian NB
    gnb = GaussianNaiveBayes()
    gnb.fit(X_train, y_train)
    
    # Make predictions
    train_accuracy = gnb.score(X_train, y_train)
    test_accuracy = gnb.score(X_test, y_test)
    
    print(f"Gaussian NB Train Accuracy: {train_accuracy:.4f}")
    print(f"Gaussian NB Test Accuracy: {test_accuracy:.4f}")
    
    # Plot feature distributions
    gnb.plot_feature_distributions(X_train, y_train)
    
    # 2D visualization
    X_2d, y_2d = make_classification(n_samples=300, n_features=2, n_redundant=0,
                                   n_informative=2, n_clusters_per_class=1, 
                                   n_classes=3, random_state=42)
    
    gnb_2d = GaussianNaiveBayes()
    gnb_2d.fit(X_2d, y_2d)
    gnb_2d.plot_decision_boundary(X_2d, y_2d)
    
    print("\n=== Multinomial Naive Bayes Example ===")
    
    # Generate discrete data for Multinomial NB (e.g., word counts)
    np.random.seed(42)
    n_samples = 300
    n_features = 5
    
    # Create synthetic "document" data with word counts
    X_multinomial = np.random.poisson(2, (n_samples, n_features))
    # Create labels based on dominant features
    y_multinomial = np.argmax(X_multinomial[:, :3], axis=1)
    
    # Split data
    split_idx = int(0.8 * len(X_multinomial))
    X_train_mult = X_multinomial[:split_idx]
    X_test_mult = X_multinomial[split_idx:]
    y_train_mult = y_multinomial[:split_idx]
    y_test_mult = y_multinomial[split_idx:]
    
    # Create and train Multinomial NB
    mnb = MultinomialNaiveBayes(alpha=1.0)
    mnb.fit(X_train_mult, y_train_mult)
    
    # Make predictions
    train_accuracy_mult = mnb.score(X_train_mult, y_train_mult)
    test_accuracy_mult = mnb.score(X_test_mult, y_test_mult)
    
    print(f"Multinomial NB Train Accuracy: {train_accuracy_mult:.4f}")
    print(f"Multinomial NB Test Accuracy: {test_accuracy_mult:.4f}")
    
    # Show some predictions with probabilities
    print("\nSample predictions with probabilities:")
    sample_probs = mnb.predict_proba(X_test_mult[:5])
    sample_preds = mnb.predict(X_test_mult[:5])
    
    for i in range(5):
        print(f"Sample {i}: Predicted={sample_preds[i]}, "
              f"Probabilities={sample_probs[i]}, Actual={y_test_mult[i]}")