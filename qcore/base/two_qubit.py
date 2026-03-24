from qcore.base.operator import Operator
import torch

class TwoQubitGate(Operator):

    def __init__(self, name, control, target, params=None):
        super().__init__(name, [control, target], params)

        self.control = control
        self.target  = target