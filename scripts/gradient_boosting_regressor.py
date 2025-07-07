
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyRegressor

class GradientBoostingRegressor:
    """
    Gradient Boosting Regressor implementation from scratch.
    
    This implementation uses scikit-learn's DecisionTreeRegressor as the weak learner.
    """
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2):
        """
        Initialize the Gradient Boosting Regressor model.
        
        Parameters:
        n_estimators (int): The number of boosting stages to perform.
        learning_rate (float): Shrinks the contribution of each tree.
        max_depth (int): Maximum depth of the individual regression estimators.
        min_samples_split (int): The minimum number of samples required to split an internal node.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.estimators_ = []
        self.initial_estimator_ = None

    def fit(self, X, y):
        """
        Train the Gradient Boosting model.
        
        Parameters:
        X (array): Training features (m x n).
        y (array): Training targets (m,).
        """
        X, y = np.array(X), np.array(y)
        
        # Step 1: Initialize model with a constant value (mean of y)
        self.initial_estimator_ = DummyRegressor(strategy='mean')
        self.initial_estimator_.fit(X, y)
        
        # Start with the initial prediction
        current_predictions = self.initial_estimator_.predict(X)
        
        # Step 2: Iteratively build trees
        for _ in range(self.n_estimators):
            # Compute residuals (pseudo-residuals), which are the negative gradient
            residuals = y - current_predictions
            
            # Fit a weak learner (decision tree) to the residuals
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=42
            )
            tree.fit(X, residuals)
            
            # Update the model's predictions
            # Add the predictions from the new tree, scaled by the learning rate
            update = self.learning_rate * tree.predict(X)
            current_predictions += update
            
            # Store the trained tree
            self.estimators_.append(tree)
            
    def predict(self, X):
        """
        Make predictions on new data.
        
        Parameters:
        X (array): Features to predict on.
        
        Returns:
        array: Predicted values.
        """
        X = np.array(X)
        
        # Start with the prediction from the initial constant estimator
        predictions = self.initial_estimator_.predict(X)
        
        # Add the predictions from each tree in the ensemble
        for tree in self.estimators_:
            predictions += self.learning_rate * tree.predict(X)
            
        return predictions

    def score(self, X, y):
        """
        Calculate the R-squared score for the model.
        
        Parameters:
        X (array): Features.
        y (array): True targets.
        
        Returns:
        float: R-squared score.
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        
        # Avoid division by zero if ss_tot is zero
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
            
        return 1 - (ss_res / ss_tot)

# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    # Generate sample regression data
    X, y = make_regression(n_samples=500, n_features=1, noise=20, 
                           random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train Gradient Boosting model
    gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
    gbr.fit(X_train, y_train)
    
    # Evaluate the model
    train_r2 = gbr.score(X_train, y_train)
    test_r2 = gbr.score(X_test, y_test)
    
    print(f"Train R-squared: {train_r2:.4f}")
    print(f"Test R-squared: {test_r2:.4f}")
    
    # Make predictions for plotting
    y_pred = gbr.predict(X_test)

    # Plot the results
    plt.figure(figsize=(10, 7))
    plt.scatter(X_test, y_test, color='black', alpha=0.5, label='Actual values')
    # Sort values for a continuous line plot
    sort_axis = np.argsort(X_test.ravel())
    plt.plot(X_test[sort_axis], y_pred[sort_axis], color='red', linewidth=2, label='Predicted values')
    plt.title("Gradient Boosting Regressor")
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.legend()
    plt.show()
