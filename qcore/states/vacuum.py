import torch

def vacuum_state(n_qubits, dtype=torch.complex64, device=None):
    dim = 2 ** n_qubits
    state = torch.zeros(dim, dtype=dtype, device=device)
    state[0] = 1.0 + 0.0j
    return state