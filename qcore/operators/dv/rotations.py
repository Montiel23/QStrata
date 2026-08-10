import numpy as np
import torch
from qcore.base.single_qubit import SingleQubitGate


class RX(SingleQubitGate):

    def __init__(self, theta, wire):
        super().__init__("RX", wire, theta)

    def matrix(self):
        theta = self.params
        if not isinstance(theta, torch.Tensor):
            theta = torch.tensor(theta)

        c = torch.cos(theta / 2.0)
        s = torch.sin(theta / 2.0)

        # dim=-1 and dim=-2 automatically handle both scalar [] and batch [batch_size] dimensions
        row1 = torch.stack([c, -1j * s], dim=-1)
        row2 = torch.stack([-1j * s, c], dim=-1)

        return torch.stack([row1, row2], dim=-2).to(
            dtype=torch.complex64, device=theta.device
        )


class RY(SingleQubitGate):

    def __init__(self, theta, wire):
        super().__init__("RY", wire, theta)

    def matrix(self):
        theta = self.params
        if not isinstance(theta, torch.Tensor):
            theta = torch.tensor(theta)

        c = torch.cos(theta / 2.0)
        s = torch.sin(theta / 2.0)

        row1 = torch.stack([c, -s], dim=-1)
        row2 = torch.stack([s, c], dim=-1)

        return torch.stack([row1, row2], dim=-2).to(
            dtype=torch.complex64, device=theta.device
        )


class RZ(SingleQubitGate):

    def __init__(self, theta, wire):
        super().__init__("RZ", wire, theta)

    def matrix(self):
        theta = self.params
        if not isinstance(theta, torch.Tensor):
            theta = torch.tensor(theta)

        e_1 = torch.exp(-1j * theta / 2.0)
        e_2 = torch.exp(1j * theta / 2.0)
        zeros = torch.zeros_like(e_1)

        row1 = torch.stack([e_1, zeros], dim=-1)
        row2 = torch.stack([zeros, e_2], dim=-1)

        return torch.stack([row1, row2], dim=-2).to(
            dtype=torch.complex64, device=theta.device
        )


class H(SingleQubitGate):

    def __init__(self, wire):
        super().__init__("H", wire)

    def matrix(self):
        inv_sqrt = 1.0 / np.sqrt(2.0)
        return torch.tensor(
            [[inv_sqrt, inv_sqrt], [inv_sqrt, -inv_sqrt]],
            dtype=torch.complex64,
        )