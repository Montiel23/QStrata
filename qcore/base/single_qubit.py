from qcore.base.operator import Operator
import torch

class SingleQubitGate(Operator):

    def __init__(self, name, wire, theta=None):
        super().__init__(name, [wire], theta)
        # self.wire = wire


    # def embed(self, U2, n_qubits):

    #     I = torch.eye(2, dtype=torch.complex64)

    #     result = None

    #     for q in range(n_qubits):

    #         if q == self.wire:
    #             current = U2
    #         else:
    #             current = I

    #         result = current if result is None else torch.kron(result