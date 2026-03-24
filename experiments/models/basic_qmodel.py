import torch
from qcore.ansatz.test_ansatz import ansatz
from qcore.backends.base import Backend
from qcore.states.vacuum import vacuum_state
from qcore.measurement.probability import measure_probability

# class Blob2QClassifier:
class TwoDQClassifier:

    def __init__(self, n_qubits, depth, alpha):

        self.n_qubits = n_qubits
        self.depth = depth
        self.alpha = alpha

        self.backend = Backend()

        # self.theta = torch.nn.Parameter(
        #     0.1 * torch.randn(depth, self.n_qubits, 3)
        # )

        self.theta = torch.nn.Parameter(
            0.1 * torch.randn(depth, 2, n_qubits, 3)
        )

    def build_circuit(self, x):
        return ansatz(
            x,
            self.theta,
            self.n_qubits,
            self.depth,
            self.alpha
        )

    def forward(self, x, measure_wire=0):

        circuit = self.build_circuit(x)

        U = self.backend.compile(circuit)

        state = vacuum_state(self.n_qubits)

        out = self.backend.run(U, state)

        p = measure_probability(out, measure_wire, self.n_qubits)

        return p.real, out
