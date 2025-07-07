"""
Adagrad (Adaptive Gradient) optimizer implementation.
Adapts learning rate based on historical gradients for each parameter.
Includes a demo on sparse gradient optimization problem with visualization.
"""

import numpy as np
import matplotlib.pyplot as plt


class Adagrad:
    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.sum_squares = {}
        
    def step(self, params, grads):
        for name, param in params.items():
            grad = grads[name]
            
            if name not in self.sum_squares:
                self.sum_squares[name] = np.zeros_like(param)
            
            self.sum_squares[name] += grad**2
            
            param -= self.learning_rate * grad / (np.sqrt(self.sum_squares[name]) + self.epsilon)
    
    def zero_grad(self):
        self.sum_squares = {}


if __name__ == "__main__":
    np.random.seed(42)
    
    def sparse_loss(x):
        mask = np.random.binomial(1, 0.3, size=x.shape)
        return np.sum(mask * x**2)
    
    def sparse_grad(x):
        mask = np.random.binomial(1, 0.3, size=x.shape)
        return 2 * mask * x
    
    x = np.random.randn(10, 1) * 5
    params = {"x": x}
    
    optimizer = Adagrad(learning_rate=1.0)
    
    print("Minimizing sparse gradient function")
    print(f"Initial x norm: {np.linalg.norm(x):.4f}")
    print(f"Initial f(x): {sparse_loss(x):.4f}\n")
    
    losses = []
    norms = []
    grad_sparsity = []
    
    for i in range(100):
        grad = sparse_grad(params["x"])
        grads = {"x": grad}
        optimizer.step(params, grads)
        
        losses.append(sparse_loss(params["x"]))
        norms.append(np.linalg.norm(params["x"]))
        grad_sparsity.append(np.mean(grad != 0))
        
        if i % 20 == 0:
            loss = sparse_loss(params["x"])
            print(f"Iteration {i}: f(x) = {loss:.6f}, ||x|| = {np.linalg.norm(params['x']):.6f}")
    
    print(f"\nFinal x norm: {np.linalg.norm(params['x']):.6f}")
    print(f"Final f(x): {sparse_loss(params['x']):.6f}")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    ax1.plot(losses, 'b-', linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Sparse Loss vs Iteration')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(norms, 'r-', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('||x||')
    ax2.set_title('Parameter Norm vs Iteration')
    ax2.grid(True, alpha=0.3)
    
    ax3.bar(range(10), params["x"].flatten(), color='blue', alpha=0.7)
    ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax3.set_xlabel('Parameter Index')
    ax3.set_ylabel('Value')
    ax3.set_title('Final Parameter Values')
    ax3.grid(True, alpha=0.3)
    
    ax4.plot(grad_sparsity, 'g-', linewidth=2)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Gradient Sparsity')
    ax4.set_title('Fraction of Non-zero Gradients')
    ax4.set_ylim(0, 1)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()