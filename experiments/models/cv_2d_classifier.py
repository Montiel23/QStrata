import torch
import torch.nn as nn
# from qcore.backends.gaussian_backend import GaussianBackend
from qcore.backends.cvBackend import GaussianBackend
from qcore.physics.symplectic import get_displacement_vector, get_beamsplitter_matrix, get_rotation_matrix
from qcore.physics.cv_measurement import realistic_homodyne_readout


class CV2DClassifier(nn.Module):
    def __init__(self, ansatz, n_classes, hbar=2.0):
        super().__init__()
        self.ansatz = ansatz
        self.n_modes = ansatz.n_modes
        self.hbar = hbar
        self.n_classes = n_classes
        self.backend = GaussianBackend(self.n_modes, hbar=hbar)

        #learnable bias for each mode to handle vacuum offset
        self.bias = nn.Parameter(torch.zeros(self.n_modes))

        #final linear mapping if n_modes != n_classes

        self.post_processing = nn.Linear(self.n_modes, self.n_classes)
        # self.post_processing = nn.Linear(self.n_modes, self.n_modes)
        

    def forward(self, x):
        # x shape: [batch_size, n_features]
        results = []
        for sample in x:
            mu, cov = self.backend.get_vacuum()

            #data encoding (displacement)
            #scaling features to alpha
            for i in range(self.n_modes):
                alpha = torch.complex(sample[i] / torch.sqrt(torch.tensor(2*self.hbar)), torch.tensor(0.0))
                mu = mu + get_displacement_vector(self.n_modes, i, alpha, hbar=self.hbar)

            #apply ansatz
            mu, cov = self.ansatz.apply(mu, cov, self.backend)

            #realistic readout
            #apply noise/efficiency 
            readout = []
            for i in range(self.n_modes):
                # pass mu, cov, and the current mode index 'i'
                m_i = realistic_homodyne_readout(mu, cov, mode=i)

                #apply the learnable bias AFTER the hardware noise simulation
                m_i = m_i + self.bias[i]
                readout.append(m_i)


                
                # m_i = (mu[2*i]**2 + cov[2*i, 2*i]) + self.bias[i]
                # #add realistic constraints
                # m_i = realistic_homodyne_readout(m_i)
                # readout.append(m_i)

            results.append(torch.stack(readout))

        #final class logits
        quantum_out = torch.stack(results)
        return self.post_processing(quantum_out)

        