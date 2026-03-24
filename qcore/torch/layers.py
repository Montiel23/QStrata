import torch.nn as nn

class QuantumLayer(nn.Module):
    def __init__(self, operator):
        super().__init__()
        self.operator = operator

    def forward(self, x):
        return self.operator.apply(x)