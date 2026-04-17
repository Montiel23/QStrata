import torch
import torch.nn as nn
from qcore.physics.symplectic import get_rotation_matrix, get_beamsplitter_matrix, get_squeezing_matrix

class GaussianVariationalAnsatz(nn.Module):
    def __init__(self, n_modes, depth):
        super().__init__()
        self.n_modes = n_modes
        self.depth = depth

        #trainable parameters for the Gaussian manifold
        self.squeezing_r = nn.Parameter(torch.randn(depth, n_modes) * 0.1)
        self.bs_theta = nn.Parameter(torch.randn(depth, n_modes - 1) * 0.1)
        self.rot_phi = nn.Parameter(torch.randn(depth, n_modes) * 0.1)


    def get_circuit_manifest(self):
        "list of gates in order for each mode"
        manifest = []
        for d in range(self.depth):
            layer = {
                "depth": d,
                "single_mode": ['S', 'R'],
                "two_mode": "BS"
            }

            manifest.append(layer)
        return manifest

    def apply(self, mu, cov, backend):
        for d in range(self.depth):
            #squeezing layer
            for i in range(self.n_modes):
                S = get_squeezing_matrix(self.n_modes, i, self.squeezing_r[d,i])
                mu, cov = backend.apply_symplectic(mu, cov, S)


            for i in range(self.n_modes - 1):
                #phase shift
                R = get_rotation_matrix(self.n_modes, i, self.rot_phi[d, i])
                mu, cov = backend.apply_symplectic(mu, cov, R)
                #mixing
                BS = get_beamsplitter_matrix(self.n_modes, i, i+1, self.bs_theta[d,i])
                mu, cov = backend.apply_symplectic(mu, cov, BS)


        return mu, cov