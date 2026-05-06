import torch
import torch.nn as nn
import numpy as np
# from qcore.backends.gaussian_backend import GaussianBackend
from qcore.backends.cvBackend import GaussianBackend
from qcore.physics.symplectic import get_displacement_vector, get_beamsplitter_matrix, get_rotation_matrix
from qcore.physics.cv_measurement import realistic_homodyne_readout


# class CV2DClassifier(nn.Module):
#     def __init__(self, ansatz, n_classes, hbar=2.0):
#         super().__init__()
#         self.ansatz = ansatz
#         self.n_modes = ansatz.n_modes
#         self.hbar = hbar
#         self.n_classes = n_classes
#         self.backend = GaussianBackend(self.n_modes, hbar=hbar)

#         #learnable bias for each mode to handle vacuum offset
#         self.bias = nn.Parameter(torch.zeros(self.n_modes))

#         #final linear mapping if n_modes != n_classes



#         #HYBRID CLASSICAL CV QNN MODEL
#         self.post_processing = nn.Linear(self.n_modes, self.n_classes)

        

#     def get_state_for_sample(self, x_sample):
#         "return final gaussian state (mu, cov) for single input after encoding and passing through ansatz"

#         self.eval()
#         with torch.no_grad():
#             # start vacuum state
#             mu, cov = self.backend.get_vacuum()

#             #data encoding (displacement)
#             if x_sample.ndim > 1:
#                 x_sample = x_sample.squeeze()

#             for i in range(self.n_modes):
#                 val = x_sample[i].item() / np.sqrt(2 * self.hbar)
                
#                 alpha_real = torch.tensor(val, dtype=torch.float32)
#                 alpha_imag = torch.tensor(0.0, dtype=torch.float32)

#                 alpha = torch.complex(alpha_real, alpha_imag)

#                 mu = mu + get_displacement_vector(self.n_modes, i, alpha, hbar=self.hbar)

#             # apply ansatz
#             mu_out, cov_out = self.ansatz.apply(mu, cov, self.backend)

#         return mu_out, cov_out
        

#     def forward(self, x):
#         # x shape: [batch_size, n_features]

#         if x.ndim == 1:
#             x = x.unsqueeze(0)
        
#         results = []
#         for sample in x:
#             mu, cov = self.backend.get_vacuum()

#             #data encoding (displacement)
#             #scaling features to alpha
#             for i in range(self.n_modes):
#                 val = sample[i].item() / np.sqrt(2 * self.hbar)

#                 alpha_real = torch.tensor(val, dtype=torch.float32)
#                 alpha_imag = torch.tensor(0.0, dtype=torch.float32)

#                 alpha = torch.complex(alpha_real, alpha_imag)

                
#                 # alpha = torch.complex(torch.tensor(val), torch.tensor(0.0))
                
#                 mu = mu + get_displacement_vector(self.n_modes, i, alpha, hbar=self.hbar)
#                 # alpha = torch.complex(sample[i] / torch.sqrt(torch.tensor(2*self.hbar)), torch.tensor(0.0))
#                 # mu = mu + get_displacement_vector(self.n_modes, i, alpha, hbar=self.hbar)

#             #apply ansatz
#             mu, cov = self.ansatz.apply(mu, cov, self.backend)

#             #realistic readout
#             #apply noise/efficiency 
#             readout = []
#             for i in range(self.n_modes):
#                 # pass mu, cov, and the current mode index 'i'
#                 m_i = realistic_homodyne_readout(mu, cov, mode=i)

#                 #apply the learnable bias AFTER the hardware noise simulation
#                 m_i = m_i + self.bias[i]
#                 readout.append(m_i)


                
#                 # m_i = (mu[2*i]**2 + cov[2*i, 2*i]) + self.bias[i]
#                 # #add realistic constraints
#                 # m_i = realistic_homodyne_readout(m_i)
#                 # readout.append(m_i)

#             results.append(torch.stack(readout))

#         #final class logits
#         quantum_out = torch.stack(results)
#         return self.post_processing(quantum_out)

        

# class CV2DClassifier(nn.Module):
#     def __init__(self, ansatz, n_classes, hbar=2.0):
#         super().__init__()
#         self.ansatz = ansatz
#         self.n_modes = ansatz.n_modes
#         self.hbar = hbar
#         self.n_classes = n_classes
#         self.backend = GaussianBackend(self.n_modes, hbar=hbar)

#         #physical "gain" of detectors (replace nn.linear)
#         self.gain = nn.Parameter(torch.ones(n_classes))
#         self.bias = nn.Parameter(torch.zeros(n_classes))


#     def _encode_and_evolve(self, sample):
#         mu, cov = self.backend.get_vacuum()
#         for i in range(self.n_modes):
#             val = (sample[i].item() * 5.0) / np.sqrt(2 * self.hbar)
#             alpha = torch.complex(torch.tensor(val, dtype=torch.float32),
#                                   torch.tensor(0.0, dtype=torch.float32))
            
#             mu = mu + get_displacement_vector(self.n_modes, i, alpha, hbar=self.hbar)


#         return self.ansatz.apply(mu, cov, self.backend)
    

#     def get_state_for_sample(self, x_sample):
#         self.eval()
#         with torch.no_grad():
#             if x_sample.ndim > 1: x_sample = x_sample.squeeze()
#             return self._encode_and_evolve(x_sample)
        
#     def forward(self, x):
#         if x.ndim == 1: x = x.unsqueeze(0)
#         batch_logits = []

#         for sample in x:
#             mu, cov = self._encode_and_evolve(sample)

#             # collect all physical readouts
#             readouts = []
#             for i in range(self.n_modes):
#                 readouts.append(realistic_homodyne_readout(mu, cov, mode=i, angle=0.0))
#                 readouts.append(realistic_homodyne_readout(mu, cov, mode=i, angle=np.pi/2))

#             #add global intensity
#             global_intensity = torch.norm(mu) / np.sqrt(self.hbar)
#             readouts.append(global_intensity)

#             #dynamic padding/slicing
#             #pad with zeros if readouts < classes
#             readouts_tensor = torch.stack(readouts)
#             if readouts_tensor.size(0) < self.n_classes:
#                 padding = torch.zeros(self.n_classes - readouts_tensor.size(0)).to(readouts_tensor.device)
#                 readouts_tensor = torch.cat([readouts_tensor, padding])

#             #slice to exact n_classes and apply learnable params
#             logits = readouts_tensor[:self.n_classes]
#             batch_logits.append(logits * self.gain + self.bias)


#         return torch.stack(batch_logits)
            

#     def get_state_for_sample(self, x_sample):
#         #return final physical gaussian state (mu, cov) for a single input
#         #used for wigner visualization and fidelity analysis

#         self.eval()
#         with torch.no_grad():
#             #start vacuum state
#             mu, cov = self.backend.get_vacuum()

#             #data encoding
#             if x_sample.ndim > 1:
#                 x_sample = x_sample.squeeze()

#             for i in range(self.n_modes):
                
#                 #mapping PCA feature to displacement alpha
#                 #scale val using the same logic as forward()
                
#                 # val = x_sample[i].item() / np.sqrt(2 * self.hbar)
#                 val = (x_sample[i].item() * 5.0) / np.sqrt(2 * self.hbar)


#                 #fix the type mismatch (float32)
#                 alpha = torch.complex(torch.tensor(val, dtype=torch.float32),
#                                       torch.tensor(0.0, dtype=torch.float32))
                
#                 mu = mu + get_displacement_vector(self.n_modes, i, alpha, hbar=self.hbar)

#             # apply variational ansatz
#             mu_out, cov_out = self.ansatz.apply(mu, cov, self.backend)

#         return mu_out, cov_out

#     def forward(self, x):
#         if x.ndim == 1:
#             x = x.unsqueeze(0)

#         # batch_results = []
#         batch_logits = []
#         for sample in x:
#             # encoding
#             mu, cov = self.backend.get_vacuum()
#             for i in range(self.n_modes):
#                 val = sample[i].item() / np.sqrt(2 * self.hbar)
#                 # alpha = torch.complex(torch.tensor(val), torch.tensor(0.0))
#                 alpha = torch.complex(torch.tensor(val, dtype=torch.float32), torch.tensor(0.0, dtype=torch.float32))
#                 mu = mu + get_displacement_vector(self.n_modes, i, alpha, hbar=self.hbar)

#             # evolution (ansatz)
#             mu, cov = self.ansatz.apply(mu, cov, self.backend)


#             #physical heterodyne readout
#             readouts = []
#             for i in range(self.n_modes):
#                 #measure X (angle theta)
#                 x_val = realistic_homodyne_readout(mu, cov, mode=i, angle=0.0)
#                 readouts.append(x_val)
#                 #measure P (angle pi/2)
#                 p_val = realistic_homodyne_readout(mu, cov, mode=i, angle=np.pi/2)
#                 readouts.append(p_val)


#             #physical global intensity (if classes exceed n_modes * 2)
#             #using the mean energy of the system as the final class indicator

#             if self.n_classes > len(readouts):
#                 global_intensity = torch.norm(mu) / np.sqrt(self.hbar)
#                 readouts.append(global_intensity)

#             # final physical mapping
#             logits = torch.stack(readouts)[:self.n_classes]
#             batch_logits.append(logits * self.gain + self.bias)

#             return torch.stack(batch_logits)

class CV2DClassifier(nn.Module):
    def __init__(self, ansatz, n_classes, hbar=2.0, encoding_multiplier=20.0):
        super().__init__()
        self.ansatz = ansatz
        self.n_modes = ansatz.n_modes
        self.hbar = hbar
        self.n_classes = n_classes
        self.encoding_multiplier = encoding_multiplier
        self.backend = GaussianBackend(self.n_modes, hbar=hbar)

        #physical gain/bias for the final classification mapping
        self.gain = nn.Parameter(torch.ones(n_classes))
        self.bias = nn.Parameter(torch.zeros(n_classes))

    def _encode_and_evolve(self, sample, target_class=None):
        #ensure sample is a tensor to avoid the numpy error
        if not isinstance(sample, torch.Tensor):
            sample = torch.tensor(sample, dtype=torch.float32)

        # #dynamic scale mapping
        # snr_boosts = {
        #     2: 2.0,
        #     5: 2.5,
        #     7: 1.5,
        #     8: 3.0
        # }

        # #default boost for easy classes
        # current_boost = 1.0

        # if target_class is not None:
        #     current_boost = snr_boosts.get(target_class.item(), 1.0)

        #feature engineering block
        X_feat = torch.tanh(sample) * 3.0

        for i in range(X_feat.shape[-1]):
            col = X_feat[..., i]
            # X_feat[..., i] = (col - col.mean()) / (col.std() + 1e-6)
            X_feat[..., i] = (col - col.mean()) / (1.0)


        "centralized vacuum -> displacement -> ansatz"
        mu, cov = self.backend.get_vacuum()

        total_disp = torch.zeros(2 * self.n_modes, device=mu.device, dtype=torch.float32)
        scale = np.sqrt(2 * self.hbar)

        for i in range(self.n_modes):
            val = (sample[i].item() * self.encoding_multiplier) / np.sqrt(2 * self.hbar)
            # val = (X_feat[i].item() * self.encoding_multiplier) / np.sqrt(2 * self.hbar)

            angle = i * np.pi / self.n_modes

            #x-displacement and p-displacement
            alpha_real = val * np.cos(angle)
            alpha_imag = val * np.sin(angle)

            total_disp[2*i] = alpha_real * scale
            total_disp[2*i+1] = alpha_imag * scale


        #apply it once to the vacuum mean
        mu = mu + total_disp

        #apply variational ansatz
        mu_out, cov_out = self.ansatz.apply(mu, cov, self.backend, encoded_input=total_disp)

        return mu_out, cov_out
    

    def forward(self, x, y=None):
        if x.ndim == 1: x = x.unsqueeze(0)
        batch_logits = []

        for sample in x:
        # for idx, sample in enumerate(x):
            # target = y[idx] if y is not None else None
            # mu, cov = self._encode_and_evolve(sample, target_class=target)
            mu, cov = self._encode_and_evolve(sample)

            #heterodyne readout: X and P for every mode
            readouts = []
            for i in range(self.n_modes):
                readouts.append(realistic_homodyne_readout(mu, cov, mode=i, angle=0.0))
                readouts.append(realistic_homodyne_readout(mu, cov, mode=i, angle=np.pi/2))

            #global intensity fallback
            readouts.append(torch.norm(mu) / np.sqrt(self.hbar))

            #dynamic padding/slicing
            res_tensor = torch.stack(readouts)
            if res_tensor.size(0) < self.n_classes:
                pad = torch.zeros(self.n_classes - res_tensor.size(0), device=x.device)
                res_tensor = torch.cat([res_tensor, pad])

            logits = res_tensor[:self.n_classes]
            batch_logits.append(logits * self.gain + self.bias)

        return torch.stack(batch_logits)
    

    def get_state_for_sample(self, x_sample):
        self.eval()
        with torch.no_grad():
            if x_sample.ndim > 1: x_sample = x_sample.squeeze()
        return self._encode_and_evolve(x_sample)
    

    #     for i in range(self.n_modes):
    #         val = (sample[i].item() * self.encoding_multiplier) / np.sqrt(2 * self.hbar)
    #         print(sample[i].item())
    #         # val = (sample[i].item() * self.encoding_multiplier)

    #         #map even mode to X and odd modes to P, creating 2d constellation
    #         angle = i * np.pi / self.n_modes

    #         real_part = torch.tensor(val * np.cos(angle), dtype=torch.float32)
    #         imag_part = torch.tensor(val * np.sin(angle), dtype=torch.float32)

    #         #create the complex displacement parameter
    #         alpha = torch.complex(real_part, imag_part)
            

    #         mu = mu + get_displacement_vector(self.n_modes, i, alpha, hbar=self.hbar)

    #     return self.ansatz.apply(mu, cov, self.backend)
    

    # def get_state_for_sample(self, x_sample):
    #     #use for wigner plots and fidelity and return pure physical state
    #     self.eval()
    #     with torch.no_grad():
    #         if x_sample.ndim > 1: x_sample = x_sample.squeeze()
    #         return self._enode_and_evolve(x_sample)
        

    # def forward(self, x):
    #     "state -> heterodyne -> logits"
    #     if x.ndim == 1: x = x.unsqueeze(0)
    #     batch_logits = []

    #     for sample in x:
    #         mu, cov = self._enode_and_evolve(sample)

    #         # physical heterodyne readout (x and p for each mode)
    #         readouts = []
    #         for i in range(self.n_modes):
    #             readouts.append(realistic_homodyne_readout(mu, cov, mode=i, angle=0.0))
    #             readouts.append(realistic_homodyne_readout(mu, cov, mode=i, angle=np.pi/2))

    #         # add global intensity
    #         global_intensity = torch.norm(mu) / np.sqrt(self.hbar)
    #         readouts.append(global_intensity)

    #         #dynamic padding and slicing to match n_classes
    #         readout_tensor = torch.stack(readouts)
    #         if readout_tensor.size(0) < self.n_classes:
    #             padding = torch.zeros(self.n_classes - readout_tensor.size(0)).to(readout_tensor.device)
    #             readout_tensor = torch.cat([readout_tensor, padding])

    #         logits = readout_tensor[:self.n_classes]
    #         batch_logits.append(logits * self.gain + self.bias)


    #     return torch.stack(batch_logits)