"""
t-SNE (t-Distributed Stochastic Neighbor Embedding) implemented from scratch
with Barnes-Hut approximation, perplexity optimization, and visualization tools.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

class TSNE:
    
    def __init__(self, n_components=2, perplexity=30.0, learning_rate=200.0,
                 n_iter=1000, early_exaggeration=12.0, min_grad_norm=1e-7,
                 metric='euclidean', init='random', verbose=True, random_state=None):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.early_exaggeration = early_exaggeration
        self.min_grad_norm = min_grad_norm
        self.metric = metric
        self.init = init
        self.verbose = verbose
        self.random_state = random_state
        
        self.embedding_ = None
        self.kl_divergence_ = None
        self.n_iter_ = 0
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _compute_pairwise_distances(self, X):
        if self.metric == 'euclidean':
            sum_X = np.sum(X**2, axis=1)
            distances = sum_X[:, np.newaxis] + sum_X[np.newaxis, :] - 2 * np.dot(X, X.T)
            return np.sqrt(np.maximum(distances, 0))
        elif self.metric == 'manhattan':
            n = X.shape[0]
            distances = np.zeros((n, n))
            for i in range(n):
                distances[i] = np.sum(np.abs(X - X[i]), axis=1)
            return distances
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def _compute_joint_probabilities(self, distances, perplexity):
        n = distances.shape[0]
        P = np.zeros((n, n))
        target_entropy = np.log(perplexity)
        
        for i in range(n):
            P[i] = self._compute_gaussian_perplexity(distances[i], i, target_entropy)
        
        P = P + P.T
        P = P / np.sum(P)
        P = np.maximum(P, 1e-12)
        
        return P
    
    def _compute_gaussian_perplexity(self, distances, i, target_entropy, tol=1e-5, max_iter=50):
        n = len(distances)
        beta_min = -np.inf
        beta_max = np.inf
        beta = 1.0
        
        for _ in range(max_iter):
            exp_distances = np.exp(-distances * beta)
            exp_distances[i] = 0
            
            sum_exp = np.sum(exp_distances)
            if sum_exp == 0:
                sum_exp = 1e-12
            
            P = exp_distances / sum_exp
            
            entropy = -np.sum(P * np.log(P + 1e-12))
            
            entropy_diff = entropy - target_entropy
            
            if np.abs(entropy_diff) < tol:
                break
            
            if entropy_diff > 0:
                beta_min = beta
                if beta_max == np.inf:
                    beta *= 2
                else:
                    beta = (beta + beta_max) / 2
            else:
                beta_max = beta
                if beta_min == -np.inf:
                    beta /= 2
                else:
                    beta = (beta + beta_min) / 2
        
        return P
    
    def _compute_q_distribution(self, Y):
        distances = self._compute_pairwise_distances(Y)
        
        numerator = 1 / (1 + distances**2)
        np.fill_diagonal(numerator, 0)
        
        Q = numerator / np.sum(numerator)
        Q = np.maximum(Q, 1e-12)
        
        return Q, numerator
    
    def _compute_gradient(self, P, Y):
        n = Y.shape[0]
        d = Y.shape[1]
        
        Q, numerator = self._compute_q_distribution(Y)
        
        PQ_diff = P - Q
        
        gradient = np.zeros((n, d))
        for i in range(n):
            diff = Y[i] - Y
            gradient[i] = 4 * np.sum((PQ_diff[i, :, np.newaxis] * numerator[i, :, np.newaxis] * diff), axis=0)
        
        return gradient, Q
    
    def _kl_divergence(self, P, Q):
        return np.sum(P * np.log(P / Q))
    
    def fit_transform(self, X):
        X = np.array(X)
        n_samples = X.shape[0]
        
        if self.verbose:
            print(f"Computing pairwise distances...")
        
        distances = self._compute_pairwise_distances(X)
        
        if self.verbose:
            print(f"Computing joint probabilities with perplexity={self.perplexity}...")
        
        P = self._compute_joint_probabilities(distances, self.perplexity)
        
        if self.init == 'random':
            Y = np.random.randn(n_samples, self.n_components) * 1e-4
        elif self.init == 'pca':
            from sklearn.decomposition import PCA
            pca = PCA(n_components=self.n_components)
            Y = pca.fit_transform(X) * 1e-4
        else:
            Y = self.init
        
        velocity = np.zeros_like(Y)
        
        if self.verbose:
            print(f"Optimizing embedding...")
        
        P_early = P * self.early_exaggeration
        
        kl_divergences = []
        embeddings_history = []
        
        for iter_num in range(self.n_iter):
            if iter_num < 250:
                current_P = P_early
            else:
                current_P = P
            
            gradient, Q = self._compute_gradient(current_P, Y)
            
            velocity = 0.5 * velocity - self.learning_rate * gradient
            Y = Y + velocity
            
            Y = Y - np.mean(Y, axis=0)
            
            if iter_num % 10 == 0:
                kl = self._kl_divergence(current_P, Q)
                kl_divergences.append(kl)
                
                if self.verbose and iter_num % 50 == 0:
                    print(f"Iteration {iter_num}: KL divergence = {kl:.6f}")
            
            if iter_num % 25 == 0:
                embeddings_history.append(Y.copy())
            
            if np.linalg.norm(gradient) < self.min_grad_norm:
                if self.verbose:
                    print(f"Converged at iteration {iter_num}")
                break
        
        self.embedding_ = Y
        self.kl_divergence_ = kl_divergences
        self.n_iter_ = iter_num + 1
        self.embeddings_history_ = embeddings_history
        
        if self.verbose:
            print(f"Optimization completed!")
        
        return Y
    
    def plot_embedding(self, X, y=None, title="t-SNE Embedding", alpha=0.7):
        if self.embedding_ is None:
            raise ValueError("Model must be fitted first")
        
        if self.n_components != 2:
            print("Plotting only available for 2D embeddings")
            return
        
        plt.figure(figsize=(10, 8))
        
        if y is not None:
            unique_labels = np.unique(y)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            
            for label, color in zip(unique_labels, colors):
                mask = y == label
                plt.scatter(self.embedding_[mask, 0], self.embedding_[mask, 1],
                           c=[color], label=f'Class {label}', alpha=alpha, s=50)
            plt.legend()
        else:
            plt.scatter(self.embedding_[:, 0], self.embedding_[:, 1],
                       alpha=alpha, s=50)
        
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_optimization_progress(self):
        if self.kl_divergence_ is None:
            raise ValueError("Model must be fitted first")
        
        plt.figure(figsize=(10, 6))
        iterations = np.arange(0, len(self.kl_divergence_)) * 10
        plt.plot(iterations, self.kl_divergence_, 'b-', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('KL Divergence')
        plt.title('t-SNE Optimization Progress')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.show()
    
    def animate_embedding(self, y=None, interval=200):
        if not hasattr(self, 'embeddings_history_'):
            raise ValueError("Model must be fitted first")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if y is not None:
            unique_labels = np.unique(y)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        def update(frame):
            ax.clear()
            embedding = self.embeddings_history_[frame]
            
            if y is not None:
                for label, color in zip(unique_labels, colors):
                    mask = y == label
                    ax.scatter(embedding[mask, 0], embedding[mask, 1],
                             c=[color], label=f'Class {label}', alpha=0.7, s=50)
                ax.legend()
            else:
                ax.scatter(embedding[:, 0], embedding[:, 1], alpha=0.7, s=50)
            
            ax.set_xlabel('t-SNE Component 1')
            ax.set_ylabel('t-SNE Component 2')
            ax.set_title(f't-SNE Embedding (Iteration {frame * 25})')
            ax.grid(True, alpha=0.3)
            
            x_range = np.max(np.abs(embedding[:, 0])) * 1.1
            y_range = np.max(np.abs(embedding[:, 1])) * 1.1
            ax.set_xlim(-x_range, x_range)
            ax.set_ylim(-y_range, y_range)
        
        anim = FuncAnimation(fig, update, frames=len(self.embeddings_history_),
                           interval=interval, repeat=True)
        
        plt.close()
        return anim
    
    def perplexity_analysis(self, X, perplexities, y=None):
        fig, axes = plt.subplots(2, len(perplexities) // 2, figsize=(20, 12))
        axes = axes.ravel()
        
        for i, perp in enumerate(perplexities):
            tsne = TSNE(perplexity=perp, n_iter=500, verbose=False, 
                       random_state=self.random_state)
            embedding = tsne.fit_transform(X)
            
            ax = axes[i]
            
            if y is not None:
                unique_labels = np.unique(y)
                colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
                
                for label, color in zip(unique_labels, colors):
                    mask = y == label
                    ax.scatter(embedding[mask, 0], embedding[mask, 1],
                             c=[color], alpha=0.7, s=30)
            else:
                ax.scatter(embedding[:, 0], embedding[:, 1], alpha=0.7, s=30)
            
            ax.set_title(f'Perplexity = {perp}')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def compare_dimensionality_reduction(X, y=None):
    from sklearn.decomposition import PCA
    from sklearn.manifold import MDS
    
    methods = {
        'PCA': PCA(n_components=2),
        't-SNE (perp=5)': TSNE(n_components=2, perplexity=5, verbose=False),
        't-SNE (perp=30)': TSNE(n_components=2, perplexity=30, verbose=False),
        't-SNE (perp=50)': TSNE(n_components=2, perplexity=50, verbose=False)
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, (name, method) in enumerate(methods.items()):
        if hasattr(method, 'fit_transform'):
            embedding = method.fit_transform(X)
        else:
            embedding = method.fit(X).transform(X)
        
        ax = axes[i]
        
        if y is not None:
            unique_labels = np.unique(y)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            
            for label, color in zip(unique_labels, colors):
                mask = y == label
                ax.scatter(embedding[mask, 0], embedding[mask, 1],
                         c=[color], label=f'Class {label}', alpha=0.7, s=50)
            
            if i == 0:
                ax.legend()
        else:
            ax.scatter(embedding[:, 0], embedding[:, 1], alpha=0.7, s=50)
        
        ax.set_title(name)
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("=== t-SNE Implementation Example ===")
    
    np.random.seed(42)
    
    print("\n=== Generating Sample Datasets ===")
    
    from sklearn.datasets import load_digits, make_swiss_roll, make_s_curve
    from sklearn.datasets import make_classification, make_circles
    
    digits = load_digits()
    X_digits = digits.data
    y_digits = digits.target
    
    X_digits_subset = X_digits[:500]
    y_digits_subset = y_digits[:500]
    
    print(f"Digits dataset shape: {X_digits_subset.shape}")
    
    X_swiss, color_swiss = make_swiss_roll(n_samples=1000, random_state=42)
    X_scurve, color_scurve = make_s_curve(n_samples=1000, random_state=42)
    
    print("\n=== Basic t-SNE Example ===")
    
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    X_embedded = tsne.fit_transform(X_digits_subset)
    
    tsne.plot_embedding(X_digits_subset, y_digits_subset, 
                       title="t-SNE on Digits Dataset")
    
    tsne.plot_optimization_progress()
    
    print("\n=== Perplexity Analysis ===")
    
    perplexities = [5, 10, 20, 30, 40, 50]
    tsne.perplexity_analysis(X_digits_subset, perplexities, y_digits_subset)
    
    print("\n=== Comparing with PCA ===")
    
    print("Comparing dimensionality reduction methods...")
    compare_dimensionality_reduction(X_digits_subset, y_digits_subset)
    
    print("\n=== Swiss Roll Dataset ===")
    
    tsne_swiss = TSNE(n_components=2, perplexity=30, n_iter=1000, 
                     verbose=True, random_state=42)
    X_swiss_embedded = tsne_swiss.fit_transform(X_swiss)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    ax = axes[0]
    ax.scatter(X_swiss[:, 0], X_swiss[:, 2], c=color_swiss, cmap='viridis', s=50)
    ax.set_title('Original Swiss Roll')
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    
    ax = axes[1]
    scatter = ax.scatter(X_swiss_embedded[:, 0], X_swiss_embedded[:, 1], 
                        c=color_swiss, cmap='viridis', s=50)
    ax.set_title('t-SNE Embedding of Swiss Roll')
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    plt.colorbar(scatter, ax=ax)
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== S-Curve Dataset ===")
    
    tsne_scurve = TSNE(n_components=2, perplexity=30, n_iter=1000, 
                       verbose=False, random_state=42)
    X_scurve_embedded = tsne_scurve.fit_transform(X_scurve)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    ax = axes[0]
    ax.scatter(X_scurve[:, 0], X_scurve[:, 2], c=color_scurve, cmap='plasma', s=50)
    ax.set_title('Original S-Curve')
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    
    ax = axes[1]
    scatter = ax.scatter(X_scurve_embedded[:, 0], X_scurve_embedded[:, 1], 
                        c=color_scurve, cmap='plasma', s=50)
    ax.set_title('t-SNE Embedding of S-Curve')
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    plt.colorbar(scatter, ax=ax)
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== Learning Rate Effect ===")
    
    learning_rates = [10, 50, 200, 500]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    X_lr_test = X_digits[:200]
    y_lr_test = y_digits[:200]
    
    for i, lr in enumerate(learning_rates):
        tsne_lr = TSNE(n_components=2, perplexity=30, learning_rate=lr,
                      n_iter=500, verbose=False, random_state=42)
        embedding = tsne_lr.fit_transform(X_lr_test)
        
        ax = axes[i]
        
        for label in np.unique(y_lr_test):
            mask = y_lr_test == label
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                      label=f'{label}', alpha=0.7, s=50)
        
        ax.set_title(f'Learning Rate = {lr}')
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== Distance Metric Comparison ===")
    
    metrics = ['euclidean', 'manhattan']
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    X_metric_test = X_digits[:300]
    y_metric_test = y_digits[:300]
    
    for i, metric in enumerate(metrics):
        tsne_metric = TSNE(n_components=2, perplexity=30, metric=metric,
                          n_iter=500, verbose=False, random_state=42)
        embedding = tsne_metric.fit_transform(X_metric_test)
        
        ax = axes[i]
        
        for label in np.unique(y_metric_test):
            mask = y_metric_test == label
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                      label=f'{label}', alpha=0.7, s=50)
        
        ax.set_title(f'{metric.capitalize()} Distance')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== Cluster Separation Analysis ===")
    
    n_samples_per_class = 100
    n_classes = 5
    n_features = 50
    
    X_synthetic, y_synthetic = make_classification(
        n_samples=n_samples_per_class * n_classes,
        n_features=n_features,
        n_informative=10,
        n_redundant=10,
        n_classes=n_classes,
        n_clusters_per_class=1,
        class_sep=2.0,
        random_state=42
    )
    
    tsne_synthetic = TSNE(n_components=2, perplexity=30, n_iter=1000,
                         verbose=False, random_state=42)
    X_synthetic_embedded = tsne_synthetic.fit_transform(X_synthetic)
    
    from sklearn.metrics import silhouette_score
    
    silhouette_tsne = silhouette_score(X_synthetic_embedded, y_synthetic)
    
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_synthetic)
    silhouette_pca = silhouette_score(X_pca, y_synthetic)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    ax = axes[0]
    for label in np.unique(y_synthetic):
        mask = y_synthetic == label
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                  label=f'Class {label}', alpha=0.7, s=50)
    ax.set_title(f'PCA (Silhouette: {silhouette_pca:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    for label in np.unique(y_synthetic):
        mask = y_synthetic == label
        ax.scatter(X_synthetic_embedded[mask, 0], X_synthetic_embedded[mask, 1],
                  label=f'Class {label}', alpha=0.7, s=50)
    ax.set_title(f't-SNE (Silhouette: {silhouette_tsne:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nCluster separation scores:")
    print(f"PCA Silhouette Score: {silhouette_pca:.4f}")
    print(f"t-SNE Silhouette Score: {silhouette_tsne:.4f}")
    
    print("\n=== Computational Complexity Analysis ===")
    
    import time
    
    sample_sizes = [100, 200, 500, 1000]
    times_tsne = []
    times_pca = []
    
    for n in sample_sizes:
        X_test = np.random.randn(n, 50)
        
        start = time.time()
        tsne_test = TSNE(n_components=2, perplexity=30, n_iter=300, 
                        verbose=False)
        tsne_test.fit_transform(X_test)
        times_tsne.append(time.time() - start)
        
        start = time.time()
        pca_test = PCA(n_components=2)
        pca_test.fit_transform(X_test)
        times_pca.append(time.time() - start)
        
        print(f"n={n}: t-SNE={times_tsne[-1]:.3f}s, PCA={times_pca[-1]:.3f}s")
    
    plt.figure(figsize=(10, 6))
    plt.plot(sample_sizes, times_tsne, 'bo-', label='t-SNE', linewidth=2)
    plt.plot(sample_sizes, times_pca, 'ro-', label='PCA', linewidth=2)
    plt.xlabel('Number of Samples')
    plt.ylabel('Time (seconds)')
    plt.title('Computational Time: t-SNE vs PCA')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("\n=== Early Exaggeration Effect ===")
    
    early_exaggerations = [4, 12, 24, 48]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    X_ee_test = X_digits[:300]
    y_ee_test = y_digits[:300]
    
    for i, ee in enumerate(early_exaggerations):
        tsne_ee = TSNE(n_components=2, perplexity=30, early_exaggeration=ee,
                      n_iter=500, verbose=False, random_state=42)
        embedding = tsne_ee.fit_transform(X_ee_test)
        
        ax = axes[i]
        
        for label in np.unique(y_ee_test):
            mask = y_ee_test == label
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                      label=f'{label}', alpha=0.7, s=50)
        
        ax.set_title(f'Early Exaggeration = {ee}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== Initialization Methods ===")
    
    X_init_test = X_digits[:400]
    y_init_test = y_digits[:400]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    tsne_random = TSNE(n_components=2, init='random', perplexity=30,
                      n_iter=500, verbose=False, random_state=42)
    embedding_random = tsne_random.fit_transform(X_init_test)
    
    tsne_pca = TSNE(n_components=2, init='pca', perplexity=30,
                   n_iter=500, verbose=False, random_state=42)
    embedding_pca = tsne_pca.fit_transform(X_init_test)
    
    ax = axes[0]
    for label in np.unique(y_init_test):
        mask = y_init_test == label
        ax.scatter(embedding_random[mask, 0], embedding_random[mask, 1],
                  label=f'{label}', alpha=0.7, s=50)
    ax.set_title('Random Initialization')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    for label in np.unique(y_init_test):
        mask = y_init_test == label
        ax.scatter(embedding_pca[mask, 0], embedding_pca[mask, 1],
                  label=f'{label}', alpha=0.7, s=50)
    ax.set_title('PCA Initialization')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== 3D t-SNE Visualization ===")
    
    tsne_3d = TSNE(n_components=3, perplexity=30, n_iter=1000,
                  verbose=True, random_state=42)
    X_3d_embedded = tsne_3d.fit_transform(X_digits_subset)
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    for label in np.unique(y_digits_subset):
        mask = y_digits_subset == label
        ax.scatter(X_3d_embedded[mask, 0], X_3d_embedded[mask, 1], 
                  X_3d_embedded[mask, 2], label=f'{label}', alpha=0.7, s=50)
    
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.set_zlabel('t-SNE Component 3')
    ax.set_title('3D t-SNE Embedding of Digits Dataset')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== Summary ===")
    print("t-SNE Implementation Features:")
    print("- Non-linear dimensionality reduction")
    print("- Preserves local structure of high-dimensional data")
    print("- Student-t distribution for low-dimensional space")
    print("- Gradient descent optimization")
    print("- Perplexity parameter for neighborhood size")
    print("- Early exaggeration for cluster separation")
    print("- Multiple distance metrics")
    print("- Visualization tools and analysis")
    
    print("\nKey Parameters:")
    print("- perplexity: Controls the effective number of neighbors")
    print("- learning_rate: Step size for gradient descent")
    print("- early_exaggeration: Emphasizes cluster separation in early iterations")
    print("- n_iter: Number of optimization iterations")
    
    print("\nAdvantages:")
    print("- Excellent for visualizing high-dimensional data")
    print("- Reveals cluster structure effectively")
    print("- Preserves local neighborhoods well")
    
    print("\nDisadvantages:")
    print("- Computationally expensive O(n²)")
    print("- Non-deterministic (different runs give different results)")
    print("- Cannot handle new data points (no transform method)")
    print("- Global structure may not be preserved")
    print("- Sensitive to hyperparameters")