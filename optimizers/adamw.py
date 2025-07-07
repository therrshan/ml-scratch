"""
AdamW optimizer implementation with decoupled weight decay.
Improves upon Adam by properly implementing weight decay regularization.
"""

import numpy as np
import matplotlib.pyplot as plt


class AdamW:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = {}
        self.v = {}
        self.t = 0
        
    def step(self, params, grads):
        self.t += 1
        
        for name, param in params.items():
            grad = grads[name]
            
            if name not in self.m:
                self.m[name] = np.zeros_like(param)
                self.v[name] = np.zeros_like(param)
            
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * grad**2
            
            m_hat = self.m[name] / (1 - self.beta1**self.t)
            v_hat = self.v[name] / (1 - self.beta2**self.t)
            
            param -= self.learning_rate * (m_hat / (np.sqrt(v_hat) + self.epsilon) + self.weight_decay * param)
    
    def zero_grad(self):
        self.m = {}
        self.v = {}


if __name__ == "__main__":
    np.random.seed(42)
    
    def regularized_loss(x, lambda_reg=0.1):
        data_loss = np.sum((x - 2)**2)
        reg_loss = lambda_reg * np.sum(x**2)
        return data_loss + reg_loss
    
    def regularized_grad(x, lambda_reg=0.1):
        data_grad = 2 * (x - 2)
        reg_grad = 2 * lambda_reg * x
        return data_grad + reg_grad
    
    x = np.random.randn(2, 1) * 3
    params = {"x": x}
    
    optimizer = AdamW(learning_rate=0.1, weight_decay=0.1)
    
    print("Minimizing regularized loss: (x-2)^2 + 0.1*x^2")
    print(f"Initial x: {x.flatten()}")
    print(f"Initial loss: {regularized_loss(x):.4f}\n")
    
    trajectory = [x.copy()]
    losses = []
    data_losses = []
    reg_losses = []
    
    for i in range(100):
        grad = regularized_grad(params["x"], lambda_reg=0)
        grads = {"x": grad}
        optimizer.step(params, grads)
        
        trajectory.append(params["x"].copy())
        total_loss = regularized_loss(params["x"])
        data_loss = np.sum((params["x"] - 2)**2)
        reg_loss = 0.1 * np.sum(params["x"]**2)
        
        losses.append(total_loss)
        data_losses.append(data_loss)
        reg_losses.append(reg_loss)
        
        if i % 20 == 0:
            print(f"Iteration {i}: loss = {total_loss:.6f}, x_mean = {np.mean(params['x']):.6f}")
    
    print(f"\nFinal x: {params['x'].flatten()}")
    print(f"Final loss: {regularized_loss(params['x']):.6f}")
    print(f"Theoretical optimum: x = 2/(1 + 0.1) = {2/1.1:.6f}")
    
    trajectory = np.array(trajectory).squeeze()
    theoretical_optimum = 2/1.1
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    x_range = np.linspace(-1, 4, 100)
    y_range = np.linspace(-1, 4, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = (X - 2)**2 + (Y - 2)**2 + 0.1 * (X**2 + Y**2)
    
    contour = ax1.contour(X, Y, Z, levels=30, alpha=0.6)
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'ro-', markersize=3, linewidth=1.5, alpha=0.7)
    ax1.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=8, label='Start')
    ax1.plot(trajectory[-1, 0], trajectory[-1, 1], 'bs', markersize=8, label='End')
    ax1.plot(theoretical_optimum, theoretical_optimum, 'k*', markersize=12, label=f'Optimum ({theoretical_optimum:.3f})')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_title('AdamW Optimization Path')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.semilogy(losses, 'b-', linewidth=2, label='Total Loss')
    ax2.semilogy(data_losses, 'g--', linewidth=2, label='Data Loss')
    ax2.semilogy(reg_losses, 'r--', linewidth=2, label='Regularization Loss')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss (log scale)')
    ax2.set_title('Loss Components vs Iteration')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3.plot(trajectory[:, 0], 'b-', linewidth=2, label='x1')
    ax3.plot(trajectory[:, 1], 'r-', linewidth=2, label='x2')
    ax3.axhline(y=theoretical_optimum, color='k', linestyle='--', alpha=0.5, label='Theoretical Optimum')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Parameter Value')
    ax3.set_title('Parameter Evolution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4.plot(np.linalg.norm(trajectory - theoretical_optimum, axis=1), 'g-', linewidth=2)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Distance to Optimum')
    ax4.set_title('Distance to Theoretical Optimum')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()