import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18
# import torchvision.models as models

from qcore.backends.cvBackend import (
GaussianBackend
)
from qcore.physics.symplectic import (
    get_beamsplitter_matrix, get_rotation_matrix, get_displacement_vector, get_squeezing_matrix
)
from qcore.physics.cv_measurement import realistic_homodyne_readout_v2

class CV_QNN_Spine_Classifier(nn.Module):

    def __init__(self, n_modes=4, depth=2, n_classes=8, hbar=1.0):
        super().__init__()
        self.n_modes = n_modes
        self.depth = depth
        self.n_classes = n_classes
        self.hbar = hbar

        self.squeezing = nn.Parameter(torch.randn(depth, n_modes) * 0.1)
        self.rotation = nn.Parameter(torch.randn(depth, n_modes) * 0.1)

        if n_modes > 1:
            self.bs_theta = nn.Parameter(
                torch.randn(depth, n_modes - 1) * 0.1
            )

        self.readout_angles = nn.Parameter(torch.zeros(n_modes))

        self.classical_head = nn.Linear(n_modes, n_classes)

    def get_gaussian_state(self, x):

        batch_size = x.size(0)
        device = x.device

        backend = GaussianBackend(
            n_modes=self.n_modes, hbar=self.hbar, device=device
        )

        mu, cov = backend.get_vacuum(batch_size=batch_size)

        #feature encoding layer
        for k in range(self.n_modes):
            mu = backend.displacement(mu,mode=k, alpha=x[:, k])

        #variational symplectic ansatz layers
        for d in range(self.depth):
            #singlemode squeezing and phase shift rotation
            for m in range(self.n_modes):
                S_sq = get_squeezing_matrix(
                    self.n_modes, m, self.squeezing[d, m], device=device
                )
                mu, cov = backend.apply_symplectic(mu, cov, S_sq)

                S_rot = get_rotation_matrix(
                    self.n_modes, m, self.rotation[d, m], device=device
                )

                mu, cov = backend.apply_symplectic(mu, cov, S_rot)

            if self.n_modes > 1:
                for m in range(self.n_modes - 1):
                    S_bs = get_beamsplitter_matrix(
                        self.n_modes,
                        m,
                        m+1,
                        self.bs_theta[d,m],
                        device=device,
                    )
                    mu, cov = backend.apply_symplectic(mu, cov, S_bs)

        return mu, cov

    def forward(self, x):
        mu, cov = self.get_gaussian_state(x)
        batch_size = x.size(0)

        readout_signals = []
        for m in range(self.n_modes):
            signal_m = realistic_homodyne_readout_v2(
                mu, cov, mode=m, angle=self.readout_angles[m]
            )
            readout_signals.append(signal_m)

        readout_tensor = torch.stack(readout_signals, dim=-1)

        logits = self.classical_head(readout_tensor)
        return logits

class HybridSpineCVModel(nn.Module):

    def __init__(self, n_modes=4, depth=2, n_classes=8, hbar=1.0):
        super().__init__()
        self.n_modes = n_modes

        #feature extraction with resnet18
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

        self.quantum_projection_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, n_modes),
            nn.Tanh()
        )

        #quantum classifier
        self.cv_quantum_classifier = CV_QNN_Spine_Classifier(
            n_modes=n_modes, depth=depth, n_classes = n_classes, hbar=hbar
        )

    def forward(self, x):
        latent = self.feature_extractor(x)
        latent = torch.flatten(latent, 1)

        quantum_ready_vector = (
            torch.pi * self.quantum_projection_head(latent)
        )

        logits = self.cv_quantum_classifier(quantum_ready_vector)
        return logits