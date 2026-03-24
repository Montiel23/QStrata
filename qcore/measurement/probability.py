import torch

def measure_probability(state, measure_wire, n_qubits):

    probs = (state.conj() * state).real
    dim = 2 ** n_qubits

    total = probs.new_zeros(())
    for idx in range(dim):
        bit = (idx >> (n_qubits - 1 - measure_wire)) & 1
        if bit == 1:
            total = total + probs[idx]

    return total

# def measure_probability(state, wire, n_qubits):

#     if n_qubits == 1:
#         return torch.abs(state[1]) ** 2

#     elif n_qubits == 2:

#         if wire == 0:
#             return torch.abs(state[2])** + torch.abs(state[3])**2

#         if wire == 1:
#             return torch.abs(state[1]) ** + torch.abs(state[3])**2