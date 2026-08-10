import torch

# =====================================================================
# 2. GAUSSIAN SYMPLECTIC BACKEND
# =====================================================================
class GaussianBackend:

    def __init__(self, n_modes, hbar=2.0, device="cpu"):
        self.n_modes = n_modes
        self.hbar = hbar
        self.device = device

    def get_vacuum(self, batch_size=1):
        """Returns vacuum displacement mu [B, 2N] and covariance matrix cov [B, 2N, 2N]."""
        mu = torch.zeros(batch_size, 2 * self.n_modes, device=self.device)
        cov_single = torch.eye(2 * self.n_modes, device=self.device) * (
            self.hbar / 2.0
        )
        cov = cov_single.unsqueeze(0).repeat(batch_size, 1, 1)
        return mu, cov

    def apply_symplectic(self, mu, cov, S):
        """Applies Gaussian gate matrix S: mu = mu @ S^T, cov = S @ cov @ S^T."""
        if S.ndim == 2:
            new_mu = torch.matmul(mu, S.T)
            new_cov = torch.matmul(S, torch.matmul(cov, S.T))
        else:
            new_mu = torch.matmul(S, mu.unsqueeze(-1)).squeeze(-1)
            new_cov = torch.matmul(S, torch.matmul(cov, S.transpose(-1, -2)))
        return new_mu, new_cov

    def displacement(self, mu, mode, alpha):
        """D(alpha) translates phase-space displacement vector mu."""
        scale = torch.sqrt(torch.tensor(2.0 * self.hbar, device=self.device))
        if torch.is_complex(alpha):
            real_part = alpha.real
            imag_part = alpha.imag
        else:
            real_part = alpha
            imag_part = torch.zeros_like(alpha)

        # Clone mu to ensure autograd tracks inline modification
        new_mu = mu.clone()
        new_mu[:, 2 * mode] = new_mu[:, 2 * mode] + real_part * scale
        new_mu[:, 2 * mode + 1] = new_mu[:, 2 * mode + 1] + imag_part * scale
        return new_mu