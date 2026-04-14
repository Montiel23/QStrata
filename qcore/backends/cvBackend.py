import torch

class GaussianBackend:
    def __init__(self, n_modes, hbar=2.0, device='cpu'):
        self.n_modes = n_modes
        self.hbar = hbar
        self.device = device

    def get_vacuum(self):
        # vacuum mean is 0, covariance is (hbar/2) * I
        mu = torch.zeros(2 * self.n_modes, device=self.device)
        cov = torch.eye(2 * self.n_modes, device=self.device) * (self.hbar / 2)
        return mu, cov

    def apply_symplectic(self, mu, cov, S):
        "applies a linear (gaussian) gate: mu = S*mu, V = S*V*S^T"
        new_mu = torch.mv(S, mu)
        new_cov = S @ cov @ S.t()
        return new_mu, new_cov

    def displacement(self, mu, mode, alpha):
        "D(alpha) is a simple translation of the  means"
        mu[2*mode] += torch.real(alpha) * torch.sqrt(torch.tensor(2 * self.hbar))
        mu[2*mode+1] += torch.img(alpha) * torch.sqrt(torch.tensor(2 * self.hbar))
        return mu