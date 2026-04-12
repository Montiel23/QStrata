import torch
from qcore.operators.dv.rotations import RX, H
from qcore.operators.dv.entanglers import CNOT
from qcore.circuit.circuit import Circuit
from qcore.states.vacuum import vacuum_state
from qcore.backends.base import Backend
from qcore.measurement.probability import measure_probability
from experiments.metrics import get_entropy

#temporary test
test_circuit = Circuit(2)
test_circuit.add(H(0))
test_circuit.add(CNOT(0, 1))


backend = Backend()
U = backend.compile(circuit=test_circuit)
state = vacuum_state(2)
out = backend.run(U, state)

# entropy = get_entropy(out, wire=0)
entropy = get_entropy(out, 2)
print(f"DEBUG: Bell state entropy: {entropy}")