from qcore.circuit.circuit import Circuit
from qcore.operators.dv.rotations import RX, RY, RZ, H
from qcore.operators.dv.entanglers import CNOT
import numpy as np

def get_ansatz_shape(n_qubits, depth):
    return (depth, 2, n_qubits, 3)


def medical_ansatz(x, theta, n_qubits, depth, alpha):
    circuit = Circuit(n_qubits)

    #initialize with superposition to start from a place with all state interaction
    for q in range(n_qubits):
        circuit.add(H(q))

    for d in range(depth):
        # non-linear data reuploading
        for q in range(n_qubits):
            val = np.arctan(x[q]) * alpha
            # circuit.add(RY(torch.atan(x[q]) * alpha, q))
            circuit.add(RY(val, q))
            circuit.add(RZ(x[q] * alpha, q))

        # variational layer
        for q in range(n_qubits):
            circuit.add(RX(theta[d, 0, q, 0], q))
            circuit.add(RY(theta[d, 0, q, 1], q))
            circuit.add(RZ(theta[d, 0, q, 2], q))

        # entanglement fuller connection
        if n_qubits > 1:
            for q in range(n_qubits):
                #standard ring structure
                circuit.add(CNOT(q, (q+1) % n_qubits))
                #add cross-connections for richer correlation
                if n_qubits > 2:
                    circuit.add(CNOT(q, (q+2) % n_qubits))

    return circuit

# def medical_ansatz(x, theta, n_qubits, depth, alpha):
#     circuit = Circuit(n_qubits)

#     for d in range(depth):
#         # re upload data at every layer
#         for q in range(n_qubits):
#             if q % 2 == 0:
#                 circuit.add(RY(x[q] * alpha, q))
#             else:
#                 circuit.add(RZ(x[q] * alpha, q))

#         for q in range(n_qubits):
#             circuit.add(RX(theta[d, 0, q, 0], q))
#             circuit.add(RY(theta[d, 0, q, 1], q))
#             circuit.add(RZ(theta[d, 0, q, 2], q))

#         if n_qubits > 1:
#             for q in range(n_qubits):
#                 circuit.add(CNOT(q, (q+1) % n_qubits))


#     return circuit

# def medical_ansatz(x, theta, n_qubits, depth, alpha):
#     circuit = Circuit(n_qubits)

#     for d in range(depth):
#         #encoding
#         for q in range(n_qubits):
#             circuit.add(RY(x[q] * alpha, q))

#         #variational
#         for q in range(n_qubits):
#             circuit.add(RX(theta[d, 0, q, 0], q))
#             circuit.add(RY(theta[d, 0, q, 1], q))
#             circuit.add(RZ(theta[d, 0, q, 2], q))

#         #entanglement ring
#         if n_qubits > 1:
#             for q in range(n_qubits):
#                 control = q
#                 target = (q + 1) % n_qubits
#                 circuit.add(CNOT(control, target))

#         # variational 2
#         for q in range(n_qubits):
#             circuit.add(RX(theta[d, 1, q, 0], q))
#             circuit.add(RY(theta[d, 1, q, 1], q))
#             circuit.add(RZ(theta[d, 1, q, 2], q))

#     return circuit