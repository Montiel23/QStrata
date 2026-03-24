from sklearn.datasets import make_circles
import torch
def make_quantum_circles(n_samples=200):
    X, y = make_circles(n_samples=n_samples, noise=0.1, factor=0.5, random_state=42)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)