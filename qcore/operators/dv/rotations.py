import torch
from qcore.base.single_qubit import SingleQubitGate
import numpy as np

class RX(SingleQubitGate):
    def __init__(self, theta, wire):
        super().__init__("RX", wire, theta)

    def matrix(self):
        theta = self.params

        c = torch.cos(theta/2)
        s = torch.sin(theta/2)

        row1 = torch.stack([c, -1j*s])
        row2 = torch.stack([-1j*s, c])


        return torch.stack([row1, row2]).to(torch.complex64)

        # return torch.tensor([
        #     [torch.cos(theta/2), -1j*torch.sin(theta/2)],
        #     [-1j*torch.sin(theta/2), torch.cos(theta/2)]
        # ], dtype=torch.complex64)



class RY(SingleQubitGate):
    def __init__(self, theta, wire):
        super().__init__("RY", wire, theta)

    def matrix(self):
        theta = self.params

        c = torch.cos(theta/2)
        s = torch.sin(theta/2)

        row1 = torch.stack([c, -s])
        row2 = torch.stack([s, c])

        return torch.stack([row1, row2]).to(torch.complex64)

        # return torch.tensor([
        #     [torch.cos(theta/2), -torch.sin(theta/2)],
        #     [torch.sin(theta/2), torch.cos(theta/2)]
        # ], dtype=torch.complex64)


class RZ(SingleQubitGate):
    def __init__(self, theta, wire):
        super().__init__("RZ", wire, theta)

    def matrix(self):

        theta = self.params

        e_1 = torch.exp(-1j*theta/2)
        e_2 = torch.exp(1j*theta/2)

        row1 = torch.stack([e_1, torch.zeros_like(e_1)])
        row2 = torch.stack([torch.zeros_like(e_1), e_2])

        return torch.stack([row1, row2]).to(torch.complex64)

        # return torch.tensor([
        #     [torch.exp(-1j*theta/2), 0],
        #     [0, torch.exp(1j*theta/2)]
        # ], dtype=torch.complex64)

class H(SingleQubitGate):
    def __init__(self, wire):
        # super().__init__("H", [wire])
        super().__init__("H", wire)

    def matrix(self):
        # define 2x2 matrix
        inv_sqrt = 1.0 / np.sqrt(2)

        return torch.tensor([
            [inv_sqrt, inv_sqrt],
            [inv_sqrt, -inv_sqrt]
        ], dtype=torch.complex64)


        # row1 = torch.stack([inv_sqrt, inv_sqrt])
        # row2 = torch.stack([inv_sqrt, -inv_sqrt])

        # return torch.stack([row1, row2]).to(torch.complex64)
