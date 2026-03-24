import torch

def compute_metrics(tp, fp, tn, fn):
    acc = (tp + tn) / (tp + tn + fp + fn)

    precision = tp / (tp + fp + 1e-8)

    recall = tp / (tp + fn + 1e-8)

    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return acc, precision, recall, f1


def get_entropy(state_vector, n_qubits):
    # reshape to treat qubit 0 as one dimension and rest as the other
    # this prepares to trace out quibts 1 through n-1
    psi_matrix = state_vector.reshape(2, -1)

    # compute reduces density matrix for qubit 0
    rho_A = torch.matmul(psi_matrix, psi_matrix.conj().t())

    # get eigenvalues (eigvalsh because rhoA is hermitian)
    eigvals = torch.linalg.eigvalsh(rho_A).real
    eigvals = torch.clamp(eigvals, min=1e-12)

    # return von neumann entropy
    return -torch.sum(eigvals * torch.log2(eigvals))