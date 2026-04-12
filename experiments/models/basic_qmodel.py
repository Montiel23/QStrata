import torch
import numpy as np
# from qcore.ansatz.test_ansatz import ansatz, get_ansatz_shape
from qcore.ansatz.medical_ansatz import medical_ansatz, get_ansatz_shape
from qcore.backends.base import Backend
from qcore.states.vacuum import vacuum_state
from qcore.measurement.probability import measure_probability

# class Blob2QClassifier:
class TwoDQClassifier(torch.nn.Module):

    def __init__(self, n_qubits, depth, alpha, n_classes=2):

        #initiliaze the parent torch.nn.Module
        super().__init__()

        self.n_qubits = n_qubits
        self.depth = depth
        self.alpha = alpha
        self.n_classes = n_classes

        self.backend = Backend()

        # self.theta = torch.nn.Parameter(
        #     0.1 * torch.randn(depth, self.n_qubits, 3)
        # )

        self.theta_shape = get_ansatz_shape(n_qubits, depth)

        # dynamic shape
        self.theta = torch.nn.Parameter(
            # 0.1 * torch.randn(depth, 2, n_qubits, 3)
            # np.pi * torch.randn(depth, 2, n_qubits, 3)
            # self.alpha  * torch.randn(depth, 2, n_qubits, 3)
            self.alpha  * torch.randn(*self.theta_shape)
        )

        # dynamic readout, map qubit expectations to class logits
        # self.readout_layer = torch.nn.Linear(n_qubits, n_classes)
        self.readout_layer = torch.nn.Linear(2**n_qubits, n_classes)


    def build_circuit(self, x):
        # return ansatz(
        #     x,
        #     self.theta,
        #     self.n_qubits,
        #     self.depth,
        #     self.alpha
        # )

        return medical_ansatz(
            x,
            self.theta,
            self.n_qubits,
            self.depth,
            self.alpha
        )

    def forward(self, x, n_classes=2, measure_wire=0):

        #build circuit 
        circuit = self.build_circuit(x)
        U = self.backend.compile(circuit)
        state = vacuum_state(self.n_qubits)
        out = self.backend.run(U, state)

        # extract z-expectation for all qubits
        # z_expectations = []
        # for q in range(self.n_qubits):
        #     p1 = measure_probability(out, q, self.n_qubits)
        #     z_exp = (1.0 - p1) - p1 # p0 - p1 range [-1, 1]
        #     z_expectations.append(z_exp)

        # z_tensor = torch.stack(z_expectations).to(x.device)

        # logits = self.readout_layer(z_tensor)

        probs = torch.abs(out)**2
        logits = self.readout_layer(probs.unsqueeze(0))
        
        # return logits, out
        return logits.view(-1), out