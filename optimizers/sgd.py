"""
Stochastic Gradient Descent (SGD) optimizer implementation.
Supports momentum and weight decay for enhanced optimization performance.
"""

import numpy as np
import matplotlib.pyplot as plt


class SGD:
    def __init__(self, learning_rate=0.01, momentum=0.0, weight_decay=0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = {}
        
    def step(self, params, grads):
        for name, param in params.items():
            grad = grads[name]
            
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param
            
            if name not in self.velocity:
                self.velocity[name] = np.zeros_like(param)
            
            self.velocity[name] = self.momentum * self.velocity[name] - self.learning_rate * grad
            param += self.velocity[name]
    
    def zero_grad(self):
        self.velocity = {}


if __name__ == "__main__":
    np.random.seed(42)
    
    x = np.random.randn(2, 1) * 5
    params = {"x": x}
    
    optimizer = SGD(learning_rate=0.1, momentum=0.9)
    
    print("Minimizing f(x) = x^T x")
    print(f"Initial x: {x.flatten()}")
    print(f"Initial f(x): {np.sum(x**2):.4f}\n")
    
    trajectory = [x.copy()]
    losses = [np.sum(x**2)]
    
    for i in range(50):
        grads = {"x": 2 * params["x"]}
        optimizer.step(params, grads)
        
        trajectory.append(params["x"].copy())
        losses.append(np.sum(params["x"]**2))
        
        if i % 10 == 0:
            loss = np.sum(params["x"]**2)
            print(f"Iteration {i}: f(x) = {loss:.6f}")
    
    print(f"\nFinal x: {params['x'].flatten()}")
    print(f"Final f(x): {np.sum(params['x']**2):.6f}")
    
    trajectory = np.array(trajectory).squeeze()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    x_range = np.linspace(-6, 6, 100)
    y_range = np.linspace(-6, 6, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = X**2 + Y**2
    
    contour = ax1.contour(X, Y, Z, levels=20, alpha=0.6)
    ax1.clabel(contour, inline=True, fontsize=8)
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'ro-', markersize=4, linewidth=1.5)
    ax1.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=8, label='Start')
    ax1.plot(trajectory[-1, 0], trajectory[-1, 1], 'bs', markersize=8, label='End')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_title('SGD Optimization Path')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.semilogy(losses, 'b-', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss (log scale)')
    ax2.set_title('Loss vs Iteration')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()