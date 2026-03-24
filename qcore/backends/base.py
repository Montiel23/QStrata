class Backend:
    def compile(self, circuit):
        # raise NotImplementedError
        return circuit.matrix()

    def run(self, compiled, state=None):
        # raise NotImplementedError
        return compiled @ state