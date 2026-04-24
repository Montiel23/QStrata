import torch
from scipy.linalg import sqrtm
import numpy as np

def compute_gaussian_fidelity(mu1, cov1, mu2, cov2, hbar=2.0):
    #compute fidelity between two multi-mode gaussian states based on bures fidelity for gaussian states

    d_mu = mu1 - mu2
    V_avg = 0.5 * (cov1 + cov2)

    #ensure numerical stability with small epsilon
    delta = 0.5 * d_mu.T @ np.linalg.inv(V_avg + np.eye(len(V_avg)) * 1e-6) @ d_mu

    #covariance shape term
    #measure how much squeezing of class A matches class B

    det_V1 = np.linalg.det(cov1)
    det_V2 = np.linalg.det(cov2)
    det_Vavg = np.linalg.det(V_avg)

    #this drops if one state is highly squeezed and the other is not

    shape_overlap = np.sqrt(np.sqrt(det_V1 * det_V2) /det_Vavg)

    #final fidelity is combined score
    fidelity = shape_overlap * np.exp(-delta)

    return np.clip(fidelity, 0.0, 1.0)

    # #displacement factor. higher displacement equals lower fidelity (better separation)
    # d_mu = mu1 - mu2
    
    # #standard gaussian fidelity involves complex matrix math, for research, we look at the overlap or bhattacharyya distance
    # delta = 0.5 * d_mu.T @ np.linalg.inv(0.5 * (cov1 + cov2)) @ d_mu

    # # covariance overlap
    # #how much the noise ellilpses overlap
    # V_sum = cov1 + cov2
    
    # #F = F_0 * exp(-delta)
    # overlap_score = np.exp(-delta)

    # return overlap_score


def analyze_state_separation(test_results, n_classes, hbar):
    # test results should contain predicted_mu, predicted_cov, true_label

    # compute class centroids (average mu and cov per class)
    class_centroids = {}
    for c in range(n_classes):
        indices = [i for i, label in enumerate(test_results['labels']) if label == c]
        class_centroids[c] = {
            'mu': np.mean([test_results['mus'][i] for i in indices], axis=0),
            'cov': np.mean([test_results['covs'][i] for i in indices], axis=0)
        }

    
    # build the fidelity matrix
    fidelity_matrix = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            fidelity_matrix[i, j] = compute_gaussian_fidelity(
                class_centroids[i]['mu'], class_centroids[i]['cov'],
                class_centroids[j]['mu'], class_centroids[j]['cov'], hbar
            )

    return fidelity_matrix


def count_quantum_resources(ansatz):
    #total trainable weights
    trainable_params = sum(p.numel() for p in ansatz.parameters() if p.requires_grad)

    #gate breakdown
    n_modes = ansatz.n_modes
    depth = ansatz.depth

    resources = {
        "Trainable_weights": trainable_params,
        "Total_modes": n_modes,
        "Total_layers": depth,
        "Single_mode_gates": (n_modes * 3) * depth,
        "Two_mode_gates": (n_modes - 1) * depth
    }

    return resources

def compute_metrics(tp, fp, tn, fn):
    acc = (tp + tn) / (tp + tn + fp + fn)

    precision = tp / (tp + fp + 1e-8)

    recall = tp / (tp + fn + 1e-8)

    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return acc, precision, recall, f1


def calculate_purity(cov, hbar=2.0):
    n_modes = cov.shape[0] // 2
    det_v = torch.det(cov)

    purity = (hbar / 2) **n_modes / torch.sqrt(det_v)

    return torch.clamp(purity, max=1.0)

# def calculate_purity(cov, n_modes):
#     "calculate purity of a Gaussian state: gamma = 1 / sqrt(det(V))"

#     det_v = torch.det(cov)
#     return 1.0 / torch.sqrt(det_v)


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