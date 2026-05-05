import torch
import torch.nn as nn
from qcore.physics.symplectic import get_rotation_matrix, get_beamsplitter_matrix, get_squeezing_matrix, get_displacement_vector

class GaussianVariationalAnsatz(nn.Module):
    def __init__(self, n_modes, depth):
        super().__init__()
        self.n_modes = n_modes
        self.depth = depth

        self.disp_alpha = nn.Parameter(torch.randn(depth, n_modes) * 0.05)
        # self.disp_alpha = nn.Parameter(torch.complex(
        #     torch.randn(depth, n_modes) * 0.05, # real part (X)
        #     torch.randn(depth, n_modes) * 0.05 # imaginary part (P)
        # ))

        self.disp_real = nn.Parameter(torch.randn(depth, n_modes) * 0.02)
        self.disp_imag = nn.Parameter(torch.randn(depth, n_modes) * 0.02)

        #trainable parameters for the Gaussian manifold
        # self.squeezing_r = nn.Parameter(torch.randn(depth, n_modes) * 0.1)

        self.squeezing_r = nn.Parameter(torch.randn(depth, n_modes) * 0.01)
        # self.squeezing_r = nn.Parameter(torch.randn(depth, n_modes) * 0.1)


        # self.bs_theta = nn.Parameter(torch.randn(depth, n_modes) * 0.1)
        # self.rot_phi = nn.Parameter(torch.randn(depth, n_modes) * 0.1)

        # self.bs_theta = nn.Parameter(torch.randn(depth, n_modes) * 0.2)
        self.bs_theta = nn.Parameter(torch.randn(depth, n_modes) * 0.01)
        self.rot_phi = nn.Parameter(torch.randn(depth, n_modes) * 0.2)


    def get_circuit_manifest(self):
        "list of gates in order for each mode"
        manifest = []
        for d in range(self.depth):
            layer = {
                "depth": d,
                "single_mode": ['D', 'S', 'R'],
                # "two_mode": "BS"
                # "two_mode": "BS"
                "two_mode": [(i, (i+1) % self.n_modes) for i in range(self.n_modes)]
                # "two_mode": [(i, i+1) for i in range(self.n_modes - 1)]
            }
            manifest.append(layer)
        return manifest

    def apply(self, mu, cov, backend, encoded_input=None):
        for d in range(self.depth):


            if encoded_input is not None:
                mu = mu + encoded_input

            #squeezing layer
            for i in range(self.n_modes):
                # alpha_complex = torch.complex(self.disp_alpha[d,i], torch.tensor(0.0).to(self.disp_alpha.device))
                alpha_complex = torch.complex(self.disp_real[d,i], self.disp_imag[d,i])
                # D = get_displacement_vector(self.n_modes, i, self.disp_alpha[d,i])
                D = get_displacement_vector(self.n_modes, i, alpha_complex, hbar=backend.hbar)
                mu = mu + D
                # mu = backend.apply_symplectic(mu, i , self.disp_alpha[d,i])
                
                S = get_squeezing_matrix(self.n_modes, i, self.squeezing_r[d,i])
                mu, cov = backend.apply_symplectic(mu, cov, S)

                R = get_rotation_matrix(self.n_modes, i, self.rot_phi[d,i])
                mu, cov = backend.apply_symplectic(mu, cov, R)


            for i in range(self.n_modes):
                m1, m2 = i, (i+1) % self.n_modes

                #rotation
                R = get_rotation_matrix(self.n_modes, m1, self.rot_phi[d,i])
                mu, cov = backend.apply_symplectic(mu, cov, R)

                #beamsplitter
                BS = get_beamsplitter_matrix(self.n_modes, m1, m2, self.bs_theta[d, i])
                mu, cov = backend.apply_symplectic(mu, cov, BS)


        return mu, cov