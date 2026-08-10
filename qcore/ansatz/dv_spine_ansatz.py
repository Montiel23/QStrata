import torch
import torch.nn as nn
import numpy as np

#clean imports from your repositorys operators directory

from qcore.operators.dv.rotations import RY, RZ
from qcore.operators.dv.entanglers import CNOT
from qcore.measurement.probability import measure_probability

class DV_QNN_Spine_Classifier(nn.Module):
    def __init__(self, n_qubits, n_classes, depth=2):
        """
        dv quantum ansatz classifier for spine lesions
        """

        super().__init__()
        self.n_qubits = n_qubits
        self.depth = depth
        self.n_classes = n_classes
        self.state_dim = 2 ** n_qubits

        #trainable weight scaling factors and phase offsets for data re-uploading
        self.weight_y = nn.Parameter(torch.randn(depth, n_qubits) * 0.1)
        self.bias_y = nn.Parameter(torch.zeros(depth, n_qubits))

        self.weight_z = nn.Parameter(torch.randn(depth, n_qubits) * 0.1)
        self.bias_z = nn.Parameter(torch.zeros(depth, n_qubits))

        #trainable classical mapping layer to scale Pauli expectations to class space
        self.classical_head = nn.Linear(n_qubits, n_classes)

    # def _get_ry_matrices(self, theta):
    #     """Constructs batch 2x2 complex RY rotation matrices: [batch_size, 2,
    #     2]."""
    #     cos_half = torch.cos(theta / 2.0)
    #     sin_half = torch.sin(theta / 2.0)

    #     # Build rows [batch_size, 2]
    #     row0 = torch.stack([cos_half, -sin_half], dim=-1)
    #     row1 = torch.stack([sin_half, cos_half], dim=-1)

    #     # Stack into [batch_size, 2, 2]
    #     mats = torch.stack([row0, row1], dim=-2)
    #     return mats.to(device=theta.device, dtype=torch.complex64)

    # def _get_rz_matrices(self, theta):
    #     """Constructs batch 2x2 complex RZ rotation matrices: [batch_size, 2,
    #     2]."""
    #     exp_neg = torch.exp(-1j * theta / 2.0)
    #     exp_pos = torch.exp(1j * theta / 2.0)
    #     zeros = torch.zeros_like(theta)

    #     row0 = torch.stack([exp_neg, zeros], dim=-1)
    #     row1 = torch.stack([zeros, exp_pos], dim=-1)

    #     mats = torch.stack([row0, row1], dim=-2)
    #     return mats.to(device=theta.device, dtype=torch.complex64)


    # def _apply_single_qubit_gate(self, state, g_mat, target_qubit):
    #     batch_size = state.size(0)

    #     #ensure g_mat has batch dimension [batch_size, 2, 2]

    #     if g_mat.ndim == 2:
    #         g_mat = g_mat.unsqueeze(0).repeat(batch_size, 1, 1)

    #     reshape_dims = [batch_size] + [2] * self.n_qubits
    #     state = state.view(*reshape_dims)

    #     axes = list(range(len(reshape_dims)))
    #     axes.pop(target_qubit + 1)
    #     axes.insert(1, target_qubit + 1)
    #     state = state.permute(*axes)

    #     state_flat = state.reshape(batch_size, 2, -1)
    #     #compute batch mat mut on gpu
    #     transformed_state = torch.bmm(g_mat, state_flat)

    #     transformed_state = transformed_state.view(
    #         *[batch_size, 2] + [2] * (self.n_qubits -1)
    #     )
    #     inv_axes = sorted(range(len(axes)), key=lambda k: axes[k])

    #     return transformed_state.permute(*inv_axes).reshape(batch_size, self.state_dim)

    def _apply_single_qubit_gate(self, state, gate_obj, target_qubit):
        """
        extracts the unitary matrix from your custom operator and contracts it against state-vector across a batch
        """

        batch_size = state.size(0)

        #fetch 2x2 complex tensor from rotation class
        g_mat = gate_obj.matrix() #shape [2, 2] or [batch, 2, 2]

        #ensure gate matrix has a batch dimension for broadcasting
        if g_mat.ndim == 2:
            g_mat = g_mat.unsqueeze(0).repeat(batch_size, 1, 1)

        
        # reshape state-vector to isolate target axes: [batch, 2, 2, ..., 2]
        reshape_dims = [batch_size] + [2] * self.n_qubits
        state = state.view(*reshape_dims)

        #permute target qubit axis to the front for matrix multiplication
        axes = list(range(len(reshape_dims)))
        axes.pop(target_qubit + 1)
        axes.insert(1, target_qubit + 1)
        state = state.permute(*axes)

        #flatten remaining states and compute tensor contraction via batch multiplication
        state_flat = state.reshape(batch_size, 2, -1)
        transformed_state = torch.bmm(g_mat, state_flat)

        #invert permutation axes back to sequential order
        transformed_state = transformed_state.view(*[batch_size, 2] + [2] * (self.n_qubits - 1))
        inv_axes = sorted(range(len(axes)), key=lambda k: axes[k])

        return transformed_state.permute(*inv_axes).reshape(batch_size, self.state_dim)
    

    def apply_entangling_layer(self, state):
        """
        apply circular ring topology of CNOT
        """

        batch_size = state.size(0)
        reshape_dims = [batch_size] + [2] * self.n_qubits

        for ctrl in range(self.n_qubits):
            tgt = (ctrl + 1) % self.n_qubits

            #replicate repository CNOT constructor
            _ = CNOT(control=ctrl, target=tgt)

            state = state.view(*reshape_dims)

            #separate components along the control qubit axis
            state_slices = torch.chunk(state, 2, dim=ctrl + 1)
            slice_0 = state_slices[0]
            slice_1 = state_slices[1]

            #flip slice_1 amplitudes along target axis to execute pauli-x swap
            slice_1_flipped = torch.flip(slice_1, dims=[tgt + 1])

            #recombine slices seamlessly
            state = torch.cat([slice_0, slice_1_flipped], dim=ctrl+1)
            state = state.reshape(batch_size, self.state_dim)

        return state

    def get_state_vector(self, x):
        batch_size = x.size(0)
        psi = torch.zeros(batch_size, self.state_dim, device=x.device, dtype=torch.complex64)
        psi[:,0] = 1.0 + 0.0j

        for d in range(self.depth):
            for i in range(self.n_qubits):
                feat = x[:, i]
                theta_y = feat * self.weight_y[d, i] + self.bias_y[d, i]
                theta_z = feat * self.weight_z[d, i] + self.bias_z[d, i]

                # ry_gate = RY(wire=i, theta=theta_y.mean())
                # rz_gate = RZ(wire=i, theta=theta_z.mean())
                ry_gate = RY(theta=theta_y, wire=i)
                rz_gate = RZ(theta=theta_z, wire=i)

                psi = self._apply_single_qubit_gate(psi, ry_gate, target_qubit=i)

                psi = self._apply_single_qubit_gate(psi, rz_gate, target_qubit=i)

            if self.n_qubits > 1:
                psi = self.apply_entangling_layer(psi)

        return psi


    def forward(self, x):
        psi = self.get_state_vector(x)
        batch_size = x.size(0)

        batch_expectations = []
        for sample_idx in range(batch_size):
            single_state = psi[sample_idx]
            qubit_expectations = []

            for i in range(self.n_qubits):
                p_one = measure_probability(
                    single_state, measure_wire=i, n_qubits=self.n_qubits
                )
                z_expectation = 1.0 - 2.0 * p_one
                if not isinstance(z_expectation, torch.Tensor):
                    z_expectation = torch.tensor(
                        z_expectation, device=psi.device, dtype=torch.float32
                    )

                else:
                    z_expectation = z_expectation.to(
                        device=psi.device, dtype=torch.float32
                    )

                qubit_expectations.append(z_expectation)

            batch_expectations.append(torch.stack(qubit_expectations))

        expectation_tensor = torch.stack(batch_expectations)
        return self.classical_head(expectation_tensor)

    
    # def forward(self, x):
    #     # x input shape: [batch_size, n_qubits] (from cnn projection head)

    #     batch_size = x.size(0)

    #     #initialize register state-vector to ground state |00...0> on the active device
    #     psi = torch.zeros(batch_size, self.state_dim, device=x.device, dtype=torch.complex64)
    #     psi[:,0] = 1.0 + 0.0j

    #     #multi-layer data re-uploading loop
    #     for d in range(self.depth):

    #         # step a: rotation encoding
    #         for i in range(self.n_qubits):
    #             feat = x[:, i]

    #             #combine feature maps with layer weights and biases
    #             theta_y = feat * self.weight_y[d, i] + self.bias_y[d, i]
    #             theta_z = feat * self.weight_z[d, i] + self.bias_z[d, i]

    #             #instantiate native operators per sample or batch
    #             #to maintain clean batch processing, we can map mean angle or evaluate sample elements
    #             #evaluating the mean or expanding inside the operator class
    #             # ry_gate = RY(qubit=i, theta=theta_y.mean())
    #             # rz_gate = RZ(qubit=i, theta=theta_z.mean())
    #             ry_gate = RY(wire=i, theta=theta_y.mean())
    #             rz_gate = RZ(wire=i, theta=theta_z.mean())

    #             psi = self._apply_single_qubit_gate(psi, ry_gate, target_qubit=i)
    #             psi = self._apply_single_qubit_gate(psi, rz_gate, target_qubit=i)


    #         if self.n_qubits > 1:
    #             psi = self.apply_entangling_layer(psi)


    #     # native measurement integration
    #     # we iterate sample-by-sample

    #     batch_expectations = []
    #     for sample_idx in range(batch_size):
    #         single_state = psi[sample_idx]
    #         qubit_expectations = []

    #         for i in range(self.n_qubits):
    #             p_one = measure_probability(single_state, measure_wire=i, n_qubits=self.n_qubits)

    #             # translate algebraically to pauli-z expectation
    #             z_expectation = 1.0 - 2.0 * p_one

    #             if not isinstance(z_expectation, torch.Tensor):
    #                 z_expectation = torch.tensor(
    #                     z_expectation, device=psi.device, dtype=torch.float32
    #                 )
    #             else:
    #                 z_expectation = z_expectation.to(device=psi.device, dtype=torch.float32)
                
    #             # qubit_expectations.append(torch.stack(qubit_expectations))
    #             qubit_expectations.append(z_expectation)

    #         batch_expectations.append(torch.stack(qubit_expectations))


    #     expectation_tensor = torch.stack(batch_expectations)

    #     return self.classical_head(expectation_tensor)


