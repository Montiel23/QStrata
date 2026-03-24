from qcore.circuit.drawer import draw_ascii
import torch

class Circuit:

    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.ops = []

    def matrix(self):
        dim = 2 ** self.n_qubits

        U = torch.eye(dim, dtype=torch.complex64)

        for op in self.ops:
            # U = op.matrix() @ U
            U = op.embed(self.n_qubits) @ U

        return U

    def __len__(self):
        return len(self.ops)

    def add(self, op):
        self.ops.append(op)

    def draw(self):
        draw_ascii(self)