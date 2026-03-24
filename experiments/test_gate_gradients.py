import torch
from qcore.operators.dv.rotations import RX
from qcore.operators.dv.entanglers import CNOT
from qcore.circuit.circuit import Circuit
from qcore.states.vacuum import vacuum_state
from qcore.backends.base import Backend
from qcore.measurement.probability import measure_probability


### 2 qubit entanglement gradient test ####

theta = torch.tensor(0.1, requires_grad=True)

n_qubits = 2
backend = Backend()

circuit = Circuit(n_qubits)
circuit.add(RX(theta, wire=0))
circuit.add(CNOT(0, 1))
circuit.draw()

U = backend.compile(circuit)

state0 = vacuum_state(n_qubits)

out = backend.run(U, state0)

p = measure_probability(out, measure_wire=1, n_qubits=n_qubits)

loss = p

loss.backward()

print("Probability:", p)
print("Gradient:", theta.grad)


#### FULL 1 QUBIT TEST ####
# theta = torch.tensor(0.3, requires_grad=True)

# n_qubits = 1
# backend = Backend()

# circuit = Circuit(n_qubits)
# circuit.add(RX(theta, wire=0))

# U = backend.compile(circuit)

# state0 = vacuum_state(n_qubits)

# out = backend.run(U, state0)

# p = measure_probability(out, measure_wire=0, n_qubits=n_qubits)

# loss = p

# loss.backward()

# print("Probability:", p)
# print("Gradient:", theta.grad)


### 1 QUBIT ROTATION GATE GRADIENT TEST ###

# gate = RX(theta, 0)
# U = gate.embed(1)

# state = torch.tensor([1, 0], dtype=torch.complex64)

# out = U @ state

# p = torch.abs(out[1])**2

# p.backward()

# print(theta.grad)

### BASIC GRADIENT TEST ###

# U = gate.matrix()

# loss = torch.real(U[0, 0])

# loss.backward()

# print("Gradient:", theta.grad)