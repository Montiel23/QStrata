from qcore.backends.cvBackend import GaussianBackend
from qcore.physics.symplectic import get_squeezing_matrix, get_displacement_vector, get_beamsplitter_matrix
import torch
import numpy as np
import torch.nn as nn

import matplotlib.pyplot as plt

def quadrature_readout(mu, cov):

    return mu[0]** 2 + cov[0,0]


#mini cv qnn model

class SimpleCVRegressor(nn.Module):
    def __init__(self, hbar=2.0):
        super().__init__()
        self.hbar = hbar
        self.backend = GaussianBackend(n_modes=1, hbar=hbar)
        #variational squeezing parameter
        self.r = nn.Parameter(torch.tensor([0.1]))

    def forward(self, x):
        mu, cov = self.backend.get_vacuum()

        #encoding: displace by input x
        #alpha = x/sqrt(2*hbar) to make D(alpha) move the mean exactly to x
        alpha = torch.complex(x / np.sqrt(2 * self.hbar), torch.tensor(0.0))
        mu = mu + get_displacement_vector(1, 0, alpha, hbar=self.hbar)

        #processing: squeeze
        S = get_squeezing_matrix(1, 0, self.r)
        mu, cov = self.backend.apply_symplectic(mu, cov, S)

        # readout
        return quadrature_readout(mu, cov)

# experiment

def run_physics_test():
    # part b: synthetic data generation (y = x²)
    X_train = torch.linspace(-2, 2, 20)
    y_train = X_train**2

    model = SimpleCVRegressor()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    criterion = nn.MSELoss()

    print("\training on y=x^2...")
    for epoch in range(101):
        optimizer.zero_grad()
        #simple loop for 1d regression
        outputs = torch.stack([model(val) for val in X_train])
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch} | loss: {loss.item():.4f} | squeezing r: {model.r.item():.4f}")


    with torch.no_grad():
        X_test = torch.linspace(-2.5, 2.5, 50)
        y_pred = torch.stack([model(val) for val in X_test])

    plt.scatter(X_train, y_train, color='red', label='target (x²)')
    plt.plot(X_test, y_pred, label='CV-QNN prediction')
    plt.xlabel("Input x (displacement)")
    plt.ylabel("<X²> readout")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    run_physics_test()
            

# def run_physics_corroboration():


    ### FIRST PHYSICS TEST
    # n_modes = 1
    # hbar = 2.0
    # backend = GaussianBackend(n_modes, hbar=hbar)
    # mu, cov = backend.get_vacuum()

    # print(f"Initial variance (vacuum): {cov[0,0].item():.2f}")

    # #apply displacement
    # # mu = mu + get_displacement_op(n_modes, 0, alpha=0.5+0j)
    # mu = mu + get_displacement_vector(n_modes, 0, alpha=0.5+0j)

    # #apply squeezing (r=0.5)
    # S = get_squeezing_matrix(n_modes, 0, r=0.5)
    # mu, cov = backend.apply_symplectic(mu, cov, S)

    # var_x = cov[0,0].item()
    # var_p = cov[1,1].item()

    # print(f"Squeezed var X: {var_x:.4f}")
    # print(f"Expanded Var P: {var_p:.4f}")
    # print(f"Uncertainty product: {var_x * var_p:.2f}")


    ### SECOND PHYSICS TEST
#     backend = GaussianBackend(n_modes=2)
#     mu, cov = backend.get_vacuum()

#     #queeze mode 0
#     S = get_squeezing_matrix(n_modes=2, mode=0, r=0.5)
#     #mean vector (2N) and covariance matrix (2N x 2N)
#     mu, cov = backend.apply_symplectic(mu, cov, S)

#     #mix mode 0 and mode 1 (entangle)
#     BS = get_beamsplitter_matrix(n_modes=2, m1=0, m2=1, theta=torch.tensor(np.pi/4))
#     mu, cov = backend.apply_symplectic(mu, cov, BS)

#     #check off-diagonal elements in covariance (inter-mode correlation)
#     correlation = cov[0, 2].item()
#     print(f"Inter-mode correlation (x0-x1): {correlation:.4f}")

#     if abs(correlation) > 0:
#         print("physics corroborated: entanglement")

# run_physics_corroboration()