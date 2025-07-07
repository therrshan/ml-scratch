"""
Principal Component Analysis implemented from scratch with variance analysis,
visualization, and compression/decompression demonstrations.
"""

import numpy as np
import matplotlib.pyplot as plt

class PCA:
    
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None
        self.n_features_ = None
        
    def fit(self, X):
        X = np.array(X)
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        cov_matrix = np.cov(X_centered.T)
        
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        self.explained_variance_ = eigenvalues
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = eigenvalues / total_variance
        
        if self.n_components is None:
            self.n_components = n_features
        else:
            self.n_components = min(self.n_components, n_features)
        
        self.components_ = eigenvectors[:, :self.n_components].T
        
        print(f"PCA fitted with {self.n_components} components")
        print(f"Explained variance ratio: {self.explained_variance_ratio_[:self.n_components]}")
        print(f"Cumulative explained variance: {np.cumsum(self.explained_variance_ratio_[:self.n_components])}")
        
    def transform(self, X):
        if self.components_ is None:
            raise ValueError("PCA must be fitted before transforming data")
        
        X = np.array(X)
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_transformed):
        if self.components_ is None:
            raise ValueError("PCA must be fitted before inverse transforming")
        
        X_transformed = np.array(X_transformed)
        return np.dot(X_transformed, self.components_) + self.mean_
    
    def explained_variance_analysis(self, threshold=0.95):
        if self.explained_variance_ratio_ is None:
            raise ValueError("PCA must be fitted first")
        
        cumulative_variance = np.cumsum(self.explained_variance_ratio_)
        n_components_suggested = np.argmax(cumulative_variance >= threshold) + 1
        
        print(f"To explain {threshold*100:.1f}% of variance, use {n_components_suggested} components")
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.bar(range(1, len(self.explained_variance_ratio_) + 1), 
                self.explained_variance_ratio_, alpha=0.7)
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Explained Variance by Component')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
        plt.axhline(y=threshold, color='r', linestyle='--', 
                   label=f'{threshold*100:.1f}% threshold')
        plt.axvline(x=n_components_suggested, color='r', linestyle='--', 
                   label=f'{n_components_suggested} components')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.title('Cumulative Explained Variance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return n_components_suggested
    
    def plot_components(self, feature_names=None, n_components_to_plot=None):
        if self.components_ is None:
            raise ValueError("PCA must be fitted first")
        
        if n_components_to_plot is None:
            n_components_to_plot = min(4, self.components_.shape[0])
        
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(self.n_features_)]
        
        plt.figure(figsize=(15, 4 * ((n_components_to_plot + 1) // 2)))
        
        for i in range(n_components_to_plot):
            plt.subplot(((n_components_to_plot + 1) // 2), 2, i + 1)
            
            component = self.components_[i]
            plt.bar(range(len(component)), component, alpha=0.7)
            plt.xlabel('Features')
            plt.ylabel('Component Weight')
            plt.title(f'Principal Component {i+1}\n'
                     f'Explained Variance: {self.explained_variance_ratio_[i]:.3f}')
            
            if len(feature_names) <= 20:
                plt.xticks(range(len(feature_names)), feature_names, rotation=45)
            
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_2d_projection(self, X, y=None, title="PCA 2D Projection"):
        if self.components_ is None:
            self.fit(X)
        
        pca_2d = PCA(n_components=2)
        X_2d = pca_2d.fit_transform(X)
        
        plt.figure(figsize=(10, 8))
        
        if y is not None:
            colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink']
            unique_labels = np.unique(y)
            
            for i, label in enumerate(unique_labels):
                mask = y == label
                plt.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                          c=colors[i % len(colors)], alpha=0.7, s=50, 
                          label=f'Class {label}')
            plt.legend()
        else:
            plt.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.7, s=50)
        
        plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.3f} explained variance)')
        plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.3f} explained variance)')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return X_2d
    
    def reconstruction_error(self, X):
        X_transformed = self.transform(X)
        X_reconstructed = self.inverse_transform(X_transformed)
        
        mse = np.mean((X - X_reconstructed)**2)
        return mse
    
    def compress_decompress_demo(self, X, n_components_list):
        if X.shape[1] != 2:
            print("Demo only works with 2D data for visualization")
            return
        
        fig, axes = plt.subplots(2, len(n_components_list) + 1, 
                                figsize=(4 * (len(n_components_list) + 1), 8))
        
        axes[0, 0].scatter(X[:, 0], X[:, 1], alpha=0.7, s=30)
        axes[0, 0].set_title('Original Data')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xlabel('Feature 1')
        axes[0, 0].set_ylabel('Feature 2')
        
        cov = np.cov(X.T)
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(eigenvecs[1, 1], eigenvecs[0, 1]))
        
        from matplotlib.patches import Ellipse
        mean = np.mean(X, axis=0)
        ellipse = Ellipse(mean, 2*np.sqrt(eigenvals[1]), 2*np.sqrt(eigenvals[0]), 
                         angle=angle, alpha=0.3, color='red')
        axes[0, 0].add_patch(ellipse)
        
        axes[1, 0].axis('off')
        
        reconstruction_errors = []
        
        for i, n_comp in enumerate(n_components_list):
            pca_comp = PCA(n_components=n_comp)
            X_transformed = pca_comp.fit_transform(X)
            X_reconstructed = pca_comp.inverse_transform(X_transformed)
            
            error = np.mean((X - X_reconstructed)**2)
            reconstruction_errors.append(error)
            
            if n_comp == 1:
                axes[0, i+1].scatter(X_transformed[:, 0], np.zeros_like(X_transformed[:, 0]), 
                                   alpha=0.7, s=30)
                axes[0, i+1].set_xlabel('PC1')
                axes[0, i+1].set_ylabel('')
            else:
                axes[0, i+1].scatter(X_transformed[:, 0], X_transformed[:, 1], 
                                   alpha=0.7, s=30)
                axes[0, i+1].set_xlabel('PC1')
                axes[0, i+1].set_ylabel('PC2')
            
            axes[0, i+1].set_title(f'Transformed ({n_comp} PC)')
            axes[0, i+1].grid(True, alpha=0.3)
            
            axes[1, i+1].scatter(X_reconstructed[:, 0], X_reconstructed[:, 1], 
                               alpha=0.7, s=30, color='orange')
            axes[1, i+1].scatter(X[:, 0], X[:, 1], alpha=0.3, s=10, color='blue')
            axes[1, i+1].set_title(f'Reconstructed ({n_comp} PC)\nMSE: {error:.4f}')
            axes[1, i+1].grid(True, alpha=0.3)
            axes[1, i+1].set_xlabel('Feature 1')
            axes[1, i+1].set_ylabel('Feature 2')
            axes[1, i+1].legend(['Reconstructed', 'Original'])
        
        plt.tight_layout()
        plt.show()
        
        plt.figure(figsize=(8, 6))
        plt.plot(n_components_list, reconstruction_errors, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Number of Components')
        plt.ylabel('Reconstruction Error (MSE)')
        plt.title('Reconstruction Error vs Number of Components')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return reconstruction_errors


if __name__ == "__main__":
    print("=== PCA Dimensionality Reduction Example ===")
    
    np.random.seed(42)
    
    n_samples = 500
    n_features = 5
    
    base_data = np.random.randn(n_samples, 2)
    
    X = np.zeros((n_samples, n_features))
    X[:, 0] = base_data[:, 0]
    X[:, 1] = base_data[:, 1]
    X[:, 2] = 0.8 * base_data[:, 0] + 0.2 * base_data[:, 1] + 0.1 * np.random.randn(n_samples)
    X[:, 3] = 0.3 * base_data[:, 0] + 0.7 * base_data[:, 1] + 0.1 * np.random.randn(n_samples)
    X[:, 4] = 0.1 * base_data[:, 0] + 0.1 * base_data[:, 1] + 0.8 * np.random.randn(n_samples)
    
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    print(f"Original data shape: {X.shape}")
    
    pca = PCA()
    X_pca = pca.fit_transform(X)
    
    print(f"Transformed data shape: {X_pca.shape}")
    
    optimal_components = pca.explained_variance_analysis(threshold=0.95)
    
    feature_names = ['Feature A', 'Feature B', 'Feature C', 'Feature D', 'Feature E']
    pca.plot_components(feature_names)
    
    X_2d = pca.plot_2d_projection(X, y, "PCA 2D Projection with Class Labels")
    
    print("\n=== Dimensionality Reduction Comparison ===")
    
    component_options = [1, 2, 3, 4, 5]
    reconstruction_errors = []
    
    for n_comp in component_options:
        pca_comp = PCA(n_components=n_comp)
        X_transformed = pca_comp.fit_transform(X)
        error = pca_comp.reconstruction_error(X)
        reconstruction_errors.append(error)
        
        print(f"{n_comp} components: "
              f"Reconstruction error = {error:.6f}, "
              f"Cumulative variance = {np.sum(pca_comp.explained_variance_ratio_):.3f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(component_options, reconstruction_errors, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Components')
    plt.ylabel('Reconstruction Error (MSE)')
    plt.title('Reconstruction Error vs Number of Components')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.show()
    
    print("\n=== Real Dataset Example: Iris-like Data ===")
    
    from sklearn.datasets import make_classification
    X_iris, y_iris = make_classification(n_samples=300, n_features=4, 
                                        n_informative=4, n_redundant=0,
                                        n_classes=3, n_clusters_per_class=1,
                                        random_state=42)
    
    print(f"Iris-like dataset shape: {X_iris.shape}")
    
    pca_iris = PCA()
    X_iris_pca = pca_iris.fit_transform(X_iris)
    
    pca_iris.explained_variance_analysis(threshold=0.9)
    
    X_iris_2d = pca_iris.plot_2d_projection(X_iris, y_iris, "Iris-like Data: PCA 2D Projection")
    
    if X_iris_pca.shape[1] >= 3:
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        colors = ['red', 'blue', 'green']
        for i, label in enumerate(np.unique(y_iris)):
            mask = y_iris == label
            ax.scatter(X_iris_pca[mask, 0], X_iris_pca[mask, 1], X_iris_pca[mask, 2],
                      c=colors[i], alpha=0.7, s=50, label=f'Class {label}')
        
        ax.set_xlabel(f'PC1 ({pca_iris.explained_variance_ratio_[0]:.3f})')
        ax.set_ylabel(f'PC2 ({pca_iris.explained_variance_ratio_[1]:.3f})')
        ax.set_zlabel(f'PC3 ({pca_iris.explained_variance_ratio_[2]:.3f})')
        ax.set_title('Iris-like Data: PCA 3D Projection')
        ax.legend()
        plt.show()
    
    print("\n=== Compression/Decompression Demo ===")
    
    mean1 = [2, 2]
    cov1 = [[2, 1.5], [1.5, 2]]
    data1 = np.random.multivariate_normal(mean1, cov1, 200)
    
    mean2 = [-1, -1]
    cov2 = [[1, -0.8], [-0.8, 1]]
    data2 = np.random.multivariate_normal(mean2, cov2, 200)
    
    X_demo = np.vstack([data1, data2])
    
    pca_demo = PCA()
    reconstruction_errors = pca_demo.compress_decompress_demo(X_demo, [1, 2])
    
    print("\n=== Feature Importance Analysis ===")
    
    pca_detailed = PCA(n_components=3)
    pca_detailed.fit(X_iris)
    
    feature_names_iris = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
    
    print("Feature contributions to Principal Components:")
    for i in range(min(3, pca_detailed.components_.shape[0])):
        print(f"\nPC{i+1} (explains {pca_detailed.explained_variance_ratio_[i]:.3f} of variance):")
        component = pca_detailed.components_[i]
        
        abs_component = np.abs(component)
        ranked_indices = np.argsort(abs_component)[::-1]
        
        for j in ranked_indices:
            print(f"  {feature_names_iris[j]}: {component[j]:.3f}")
    
    print("\n=== Inverse Transform Demo ===")
    
    pca_2d = PCA(n_components=2)
    X_iris_2d = pca_2d.fit_transform(X_iris)
    X_iris_reconstructed = pca_2d.inverse_transform(X_iris_2d)
    
    reconstruction_error = np.mean((X_iris - X_iris_reconstructed)**2)
    print(f"Reconstruction error with 2 components: {reconstruction_error:.6f}")
    
    print("\nOriginal vs Reconstructed (first 5 samples):")
    print("Original:")
    print(X_iris[:5])
    print("Reconstructed:")
    print(X_iris_reconstructed[:5])
    print("Difference:")
    print(X_iris[:5] - X_iris_reconstructed[:5])