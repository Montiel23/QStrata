import torch
import torch.nn as nn
from qcore.physics.symplectic import get_rotation_matrix, get_beamsplitter_matrix, get_squeezing_matrix, get_displacement_vector
import numpy as np


class GaussianVariationalAnsatz(nn.Module):
    def __init__(self, n_modes, depth, squeezing_cap=1.5):
        super().__init__()
        self.n_modes = n_modes
        self.depth = depth

        #parameter layer-by-layer initialization
        #local trainable displacement offsets
        self.disp_real = nn.Parameter(torch.randn(depth, n_modes) * 0.02)
        self.disp_imag = nn.Parameter(torch.randn(depth, n_modes) * 0.02)

        #squeezing initialized near zero and bounded later
        self.squeezing_raw = nn.Parameter(torch.randn(depth, n_modes) * 0.01)
        self.squeezing_cap = squeezing_cap

        #beam splitter mixing parameters for the cascading ring topology
        self.bs_theta = nn.Parameter(torch.randn(depth, n_modes) * 2 * np.pi)

        # phase rotations initialized across a distribution [0, 2pi]
        self.rot_phi = nn.Parameter(torch.rand(depth, n_modes)* 2 * np.pi)

    def get_circuit_manifest(self):
        manifest = []
        for d in range(self.depth):
            layer = {
                "depth": d,
                "single_mode": ['D', 'S', 'R'],
                "two_mode": [(i, (i+1) % self.n_modes) for i in range(self.n_modes)]
            }

            manifest.append(layer)
        return manifest
    
    def apply(self, mu, cov, backend, encoded_input=None):

        """
        evolve first and second moments through multi-layer re-uploading
        """

        for d in range(self.depth):

            #layer-by-layer data reuploading
            if encoded_input is not None:
                mu = mu + encoded_input

            # single-mode gates layer
            for i in range(self.n_modes):
                alpha_complex = torch.complex(self.disp_real[d, i], self.disp_imag[d, i])
                D = get_displacement_vector(self.n_modes, i, alpha_complex, hbar=backend.hbar)
                mu = mu + D.to(mu.device)

                # parameter bound squeezing
                bounded_squeezing = torch.tanh(self.squeezing_raw[d, i]) * self.squeezing_cap
                S = get_squeezing_matrix(self.n_modes, i, bounded_squeezing)
                mu, cov = backend.apply_symplectic(mu, cov, S)

                # local phase rotation
                R = get_rotation_matrix(self.n_modes, i, self.rot_phi[d, i])
                mu, cov = backend.apply_symplectic(mu, cov, R)

            
            # circular entanglement
            for i in range(self.n_modes):
                m1, m2 = i, (i+1) % self.n_modes

                #phase realignment rotation within the interferometric
                R = get_rotation_matrix(self.n_modes, m1, self.rot_phi[d, i])
                mu, cov = backend.apply_symplectic(mu, cov, R)

                #linear optical interference via Beamsplitter 
                BS = get_beamsplitter_matrix(self.n_modes, m1, m2, self.bs_theta[d, i])
                mu, cov = backend.apply_symplectic(mu, cov, BS)

        return mu, cov

