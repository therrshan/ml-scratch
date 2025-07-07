"""
Adam (Adaptive Moment Estimation) optimizer implementation.
Combines adaptive learning rates with momentum for efficient optimization.
Includes a demo on Rosenbrock function minimization with visualization.
"""

import numpy as np
import matplotlib.pyplot as plt


class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
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
            
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
    
    def zero_grad(self):
        self.m = {}
        self.v = {}


if __name__ == "__main__":
    np.random.seed(42)
    
    def rosenbrock(x):
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
    
    def rosenbrock_grad(x):
        dx0 = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
        dx1 = 200 * (x[1] - x[0]**2)
        return np.array([dx0, dx1])
    
    x = np.array([-1.0, 1.0])
    params = {"x": x}
    
    optimizer = Adam(learning_rate=0.01)
    
    print("Minimizing Rosenbrock function")
    print(f"Initial x: {x}")
    print(f"Initial f(x): {rosenbrock(x):.4f}\n")
    
    trajectory = [x.copy()]
    losses = [rosenbrock(x)]
    
    for i in range(1000):
        grad = rosenbrock_grad(params["x"])
        grads = {"x": grad}
        optimizer.step(params, grads)
        
        trajectory.append(params["x"].copy())
        losses.append(rosenbrock(params["x"]))
        
        if i % 200 == 0:
            loss = rosenbrock(params["x"])
            print(f"Iteration {i}: f(x) = {loss:.6f}, x = {params['x']}")
    
    print(f"\nFinal x: {params['x']}")
    print(f"Final f(x): {rosenbrock(params['x']):.6f}")
    print(f"Optimum at: [1, 1]")
    
    trajectory = np.array(trajectory)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    x_range = np.linspace(-1.5, 1.5, 400)
    y_range = np.linspace(-0.5, 1.5, 400)
    X, Y = np.meshgrid(x_range, y_range)
    Z = (1 - X)**2 + 100 * (Y - X**2)**2
    
    levels = np.logspace(-1, 3, 20)
    contour = ax1.contour(X, Y, Z, levels=levels, alpha=0.6)
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'ro-', markersize=3, linewidth=1.5, alpha=0.7)
    ax1.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=8, label='Start')
    ax1.plot(1, 1, 'k*', markersize=15, label='Optimum')
    ax1.plot(trajectory[-1, 0], trajectory[-1, 1], 'bs', markersize=8, label='End')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_title('Adam on Rosenbrock Function')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-0.5, 1.5)
    
    ax2.semilogy(losses, 'b-', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss (log scale)')
    ax2.set_title('Loss vs Iteration')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()