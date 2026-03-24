import torch
from qcore.base.two_qubit import TwoQubitGate

class CNOT(TwoQubitGate):

    def __init__(self, control, target):
        super().__init__("CNOT", control, target)

    def matrix(self):
        return torch.tensor([
            [1,0,0,0],
            [0,1,0,0],
            [0,0,0,1],
            [0,0,1,0]
        ], dtype=torch.complex64)


class CZ(TwoQubitGate):
    def __init__(self, control, target):
        super().__init__("CZ", control, target)

    def matrix(self):

        return torch.tensor([
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,-1]
        ], dtype=torch.complex64)


class SWAP(TwoQubitGate):

    def __init__(self, q0, q1):
        super().__init__("SWAP", q0, q1)

    def matrix(self):

        return torch.tensor([
            [1,0,0,0],
            [0,0,1,0],
            [0,1,0,0],
            [0,0,0,1]
        ], dtype=torch.complex64)