"""
RMSprop optimizer implementation.
Uses adaptive learning rates based on moving average of squared gradients.
"""

import numpy as np
import matplotlib.pyplot as plt


class RMSprop:
    def __init__(self, learning_rate=0.01, alpha=0.99, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.epsilon = epsilon
        self.square_avg = {}
        
    def step(self, params, grads):
        for name, param in params.items():
            grad = grads[name]
            
            if name not in self.square_avg:
                self.square_avg[name] = np.zeros_like(param)
            
            self.square_avg[name] = self.alpha * self.square_avg[name] + (1 - self.alpha) * grad**2
            
            param -= self.learning_rate * grad / (np.sqrt(self.square_avg[name]) + self.epsilon)
    
    def zero_grad(self):
        self.square_avg = {}


if __name__ == "__main__":
    np.random.seed(42)
    
    def noisy_quadratic(x):
        return np.sum(x**2) + 0.1 * np.random.randn()
    
    def noisy_quadratic_grad(x):
        return 2 * x + 0.5 * np.random.randn(*x.shape)
    
    x = np.random.randn(2, 1) * 3
    params = {"x": x}
    
    optimizer = RMSprop(learning_rate=0.1)
    
    print("Minimizing noisy quadratic function")
    print(f"Initial x: {x.flatten()}")
    print(f"Initial f(x): {noisy_quadratic(x):.4f}\n")
    
    trajectory = [x.copy()]
    losses = []
    true_losses = []
    
    for i in range(100):
        grad = noisy_quadratic_grad(params["x"])
        grads = {"x": grad}
        optimizer.step(params, grads)
        
        trajectory.append(params["x"].copy())
        loss = noisy_quadratic(params["x"])
        losses.append(loss)
        true_losses.append(np.sum(params["x"]**2))
        
        if i % 20 == 0:
            print(f"Iteration {i}: f(x) = {loss:.6f}")
    
    print(f"\nFinal x: {params['x'].flatten()}")
    print(f"Final f(x): {np.mean(losses[-10:]):.6f} (averaged over last 10 iterations)")
    
    trajectory = np.array(trajectory).squeeze()
    
    fig = plt.figure(figsize=(15, 5))
    
    ax1 = plt.subplot(131)
    x_range = np.linspace(-4, 4, 100)
    y_range = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = X**2 + Y**2
    
    contour = ax1.contour(X, Y, Z, levels=20, alpha=0.6)
    ax1.clabel(contour, inline=True, fontsize=8)
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'ro-', markersize=3, linewidth=1.5, alpha=0.7)
    ax1.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=8, label='Start')
    ax1.plot(trajectory[-1, 0], trajectory[-1, 1], 'bs', markersize=8, label='End')
    ax1.plot(0, 0, 'k*', markersize=10, label='Optimum')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_title('RMSprop Optimization Path')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(132)
    ax2.plot(losses, 'b-', alpha=0.5, label='Noisy Loss')
    ax2.plot(true_losses, 'r-', linewidth=2, label='True Loss')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss vs Iteration')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = plt.subplot(133)
    ax3.semilogy(true_losses, 'r-', linewidth=2)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('True Loss (log scale)')
    ax3.set_title('True Loss vs Iteration (Log Scale)')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()