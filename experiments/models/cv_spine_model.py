import torch
import torch.nn as nn
import numpy as np
from qcore.backends.cvBackend import GaussianBackend
from qcore.physics.cv_measurement import realistic_homodyne_readout

class SpineCVQNN(nn.Module):
    def __init__(self, ansatz, n_classes):
        super().__init__()
        self.ansatz = ansatz
        self.n_modes = ansatz.n_modes
        self.n_classes = n_classes
        self.encoding_multiplier = self.encoding_multiplier
        self.backend = GaussianBackend(self.n_modes)

        self.gain = nn.Parameter(torch.ones(n_classes))
        self.bias = nn.Parameter(torch.zeros(n_classes))

        #readout control
        self.readout_mode = "dual_homodyne"

    def _encode_and_evolve(self, sample):
        
        if not isinstance(sample, torch.Tensor):
            sample = torch.tensor(sample, dtype=torch.float32)


        #feature engineering block???
        #X_feat = torch.tanh(sample) * 3.0

        #for i in range(X_feat.shape[-1]):
            #col = X_feat[..., i]
            # X_feat[..., i] = (col - col.mean()) / (1.0)

        #centralized vacuum -> displacement -> ansatz
        mu, cov = self.backend.get_vacuum()
        hbar = 1.0

        total_disp = torch.zeros(2 * self.n_modes, device=mu.device, dtype=torch.float32)
        scale = np.sqrt(2 * self.hbar)

        for i in range(self.n_modes):
            val = (sample[i].item() * self.encoding_multiplier) / np.sqrt(2 * self.hbar)

            angle = i * np.pi / self.n_modes

            #x and p displacement
            alpha_real = val * np.cos(angle)
            alpha_imag = val * np.sin(angle)

            total_disp[2*i] = alpha_real * scale
            total_disp[2*i+1] = alpha_imag * scale

        #apply to the vacuum mean
        mu = mu + total_disp

        #apply variational ansatz
        mu_out, cov_out = self.ansatz.apply(mu, cov, self.backend, encoded_input = total_disp)

        return mu_out, cov_out
        

    def forward(self, x, readout_type="dual_homodyne"):
        if x.ndim == 1: x = x.unsqueeze(0)
        batch_logits = []

        for sample in x:
            mu, cov = self._encode_and_evolve(sample)
            readouts = []

            if readout_type == "homodyne":
                for i in range(self.n_modes):
                    readouts.append(realistic_homodyne_readout(mu, cov, mode=i, angle=0.0))


            elif readout_type == "dual_homodyne":
                for i in range(self.n_modes):
                    readouts.append(realistic_homodyne_readout(mu, cov, mode=i, angle=0.0))
                    readouts.append(realistic_homodyne_readout(mu, cov, mode=i, angle=np.pi/2))


            res_tensor = torch.stack(readouts)

            if res_tensor.size(0) < self.n_classes:
                pad = torch.zeros(self.n_classes - res_tensor.size(0), device=x.device)
                res_tensor = torch.cat([res_tensor, pad])

            logits = res_tensor[:self.n_classes]

            batch_logits.append(logits)

        return torch.stack(batch_logits)
    
    def get_state_for_sample(self, x_sample):
        self.eval()
        with torch.no_grad():
            if x_sample.ndim > 1: x_sample = x_sample.squeeze()
        return self._encode_and_evolve(x_sample)