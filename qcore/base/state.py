class QuantumState:
    def __init__(self, vector):
        self.vector = vector

    def norm(self):
        return (self.vector.conj() * self.vector).sum()