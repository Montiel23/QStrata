from qcore.circuit.circuit import Circuit
from qcore.operators.dv.rotations import RX, RY, RZ, H
from qcore.operators.dv.entanglers import CNOT
import math

def ansatz(x, theta, n_qubits, depth, alpha):
    circuit = Circuit(n_qubits)

    for d in range(depth):

        #data encoding
        for q in range(n_qubits):
            feature_idx = q % len(x)
            circuit.add(RY(x[feature_idx] * alpha, q))

        #add hadamard gate for entanglement
        for q in range(n_qubits):
            circuit.add(H(q))

        
        #variational layer
        for q in range(n_qubits):
            circuit.add(RX(theta[d, 0, q, 0], q))
            circuit.add(RY(theta[d, 0, q, 1], q))
            circuit.add(RZ(theta[d, 0, q, 2], q))

        #scalable entanglement
        if n_qubits > 1:
            for q in range(n_qubits):
                control = q
                target = (q + 1) % n_qubits
                circuit.add(CNOT(control, target))

        #MISSING VARIATIONAL LAYER

        for q in range(n_qubits):
            circuit.add(RX(theta[d, 1, q, 0], q))
            circuit.add(RY(theta[d, 1, q, 1], q))
            circuit.add(RZ(theta[d, 1, q, 2], q))


    return circuit
        
            

# def ansatz(x, theta, n_qubits, depth):
#     circuit = Circuit(n_qubits)

#     # encoding

#     # circuit.add(RY(x[0] * math.pi, 0))
#     # circuit.add(RY(x[1] * math.pi, 1))

#     alpha = 0.5


#     #circuit.add(RY(x[0], 0))
#     #circuit.add(RY(x[1], 1))

#     circuit.add(RY(x[0] * alpha, 0))
#     circuit.add(RY(x[1] * alpha, 1))


#     for d in range(depth):

        
#         # variational layer
#         for q in range(n_qubits):
#             # parameterized gates
#             circuit.add(RX(theta[d, 0, q, 0],q))
#             circuit.add(RY(theta[d, 0, q, 1],q))
#             circuit.add(RZ(theta[d, 0, q, 2],q))


            
#             # circuit.add(RX(theta[d,q,0],q))
#             # circuit.add(RY(theta[d,q,1],q))
#             # circuit.add(RZ(theta[d,q,2],q))



#         # entanglement
#         circuit.add(CNOT(0, 1))


#         # variational layer
#         for q in range(n_qubits):
#             # parameterized gates
#             circuit.add(RX(theta[d, 1, q, 0],q))
#             circuit.add(RY(theta[d, 1, q, 1],q))
#             circuit.add(RZ(theta[d, 1, q, 2],q))
            

#         # # variational layer
#         # for q in range(n_qubits):
#         #     # parameterized gates
#         #     circuit.add(RX(theta[d, q, 0], q))
#         #     circuit.add(RY(theta[d, q, 1], q))
#         #     circuit.add(RZ(theta[d, q, 2], q))

#     return circuit
