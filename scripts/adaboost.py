
import numpy as np
from sklearn.tree import DecisionTreeClassifier

class AdaBoost:
    """
    AdaBoost (Adaptive Boosting) classifier implementation from scratch.
    
    This implementation uses scikit-learn's DecisionTreeClassifier as the weak learner.
    """
    
    def __init__(self, n_estimators=50, learning_rate=1.0):
        """
        Initialize the AdaBoost model.
        
        Parameters:
        n_estimators (int): The number of weak learners to train.
        learning_rate (float): Shrinks the contribution of each classifier.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.estimators = []
        self.estimator_weights = []
        self.classes_ = None
        
    def fit(self, X, y):
        """
        Train the AdaBoost model.
        
        Parameters:
        X (array): Training features (m x n).
        y (array): Training labels (m,).
        """
        X, y = np.array(X), np.array(y)
        self.classes_ = np.unique(y)
        n_samples = X.shape[0]
        
        # Initialize sample weights
        sample_weights = np.full(n_samples, (1 / n_samples))
        
        for _ in range(self.n_estimators):
            # Train a weak learner (stump)
            # A stump is a decision tree with max_depth=1
            estimator = DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2)
            
            # Resample data with replacement based on sample weights
            # Note: sklearn's fit method supports sample_weight directly
            estimator.fit(X, y, sample_weight=sample_weights)
            
            # Make predictions
            predictions = estimator.predict(X)
            
            # Calculate weighted error
            # We use a mask for incorrect predictions
            incorrect_mask = (predictions != y)
            error = np.sum(sample_weights[incorrect_mask])
            
            # Avoid division by zero
            if error == 0:
                error = 1e-10
            
            # Calculate estimator weight (alpha)
            estimator_weight = self.learning_rate * np.log((1 - error) / error)
            
            # Update sample weights
            # Increase weights for misclassified samples
            sample_weights *= np.exp(estimator_weight * incorrect_mask)
            
            # Normalize sample weights
            sample_weights /= np.sum(sample_weights)
            
            # Store the trained estimator and its weight
            self.estimators.append(estimator)
            self.estimator_weights.append(estimator_weight)
            
    def predict(self, X):
        """
        Make predictions on new data.
        
        Parameters:
        X (array): Features to predict on.
        
        Returns:
        array: Predicted class labels.
        """
        X = np.array(X)
        
        # Get predictions from each weak learner
        estimator_preds = np.array([estimator.predict(X) for estimator in self.estimators])
        
        # Combine predictions using weighted voting
        # Create an empty array to store the final predictions
        final_predictions = np.zeros(X.shape[0], dtype=self.classes_.dtype)
        
        for i, c in enumerate(self.classes_):
            # Sum the weights of estimators that predict class 'c'
            class_votes = np.sum((estimator_preds == c) * self.estimator_weights[:, np.newaxis], axis=0)
            final_predictions[np.argmax(class_votes)] = c
            
        # A more robust way to do the final prediction
        # For each sample, sum the weights for each class prediction
        scores = np.zeros((X.shape[0], len(self.classes_)))
        for i, estimator in enumerate(self.estimators):
            preds = estimator.predict(X)
            for j, class_val in enumerate(self.classes_):
                scores[:, j] += self.estimator_weights[i] * (preds == class_val)

        # Return the class with the highest score
        return self.classes_[np.argmax(scores, axis=1)]

    def score(self, X, y):
        """
        Calculate accuracy score.
        
        Parameters:
        X (array): Features.
        y (array): True labels.
        
        Returns:
        float: Accuracy score.
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)

# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    # Generate sample data
    X, y = make_classification(n_samples=500, n_features=2, n_informative=2, n_redundant=0,
                               n_clusters_per_class=1, n_classes=2, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train AdaBoost model
    adaboost = AdaBoost(n_estimators=50, learning_rate=0.5)
    adaboost.fit(X_train, y_train)
    
    # Evaluate the model
    train_accuracy = adaboost.score(X_train, y_train)
    test_accuracy = adaboost.score(X_test, y_test)
    
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Plot decision boundary
    def plot_decision_boundary(clf, X, y, title):
        plt.figure(figsize=(10, 7))
        
        # Create a mesh to plot in
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                               np.arange(y_min, y_max, 0.02))
        
        # Predict on the mesh
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot the contour and training examples
        plt.contourf(xx, yy, Z, alpha=0.3)
        plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
        plt.title(title)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()

    plot_decision_boundary(adaboost, X, y, "AdaBoost Decision Boundary")
