class Displacement(QuantumOperator):
    def matrix(self):
        raise NotImplementedError("Backend-specific implementation required")


    def apply(self, state):
        raise NotImplementedError