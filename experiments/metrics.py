import torch
from scipy.linalg import sqrtm
import numpy as np

def compute_gaussian_fidelity(mu1, cov1, mu2, cov2, hbar=2.0):
    #compute fidelity between two multi-mode gaussian states based on bures fidelity for gaussian states

    mu1, cov1 = np.array(mu1, dtype=np.float64), np.array(cov1, dtype=np.float64)
    mu2, cov2 = np.array(mu2, dtype=np.float64), np.array(cov2, dtype=np.float64)

    mu1 = mu1.flatten()
    mu2 = mu2.flatten()

    d_mu = mu1 - mu2
    V_avg = 0.5 * (cov1 + cov2)

    V_inv = np.linalg.inv(V_avg + np.eye(len(V_avg)) * 1e-7)

    d_mu_64 = d_mu.astype(np.float64)

    # print(f"DEBUG: MU1 NORM: {np.linalg.norm(mu1)}")
    # print(f"DEBUG: MU2 NORM: {np.linalg.norm(mu2)}")

    #ensure numerical stability with small epsilon
    # delta = 0.5 * d_mu.T @ np.linalg.inv(V_avg + np.eye(len(V_avg)) * 1e-6) @ d_mu
    delta = 0.5 * d_mu.T @ V_inv @ d_mu_64
    # delta = (1.0 / (2 * hbar)) * d_mu.T @ V_inv @ d_mu_64

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


def apply_quantum_scaling(subset):
    X, y = subset
    X_tensor = torch.tensor(X, dtype=torch.float32)

    # non-linear squashing
    X_trans = torch.tanh(X_tensor) * 3.0

    #mean centering to the vacuum center
    for i in range(X_trans.shape[-1]):
        col = X_trans[..., i]
        X_trans[..., i] = col - col.mean()

    return X_trans.numpy(), y

def analyze_state_separation(test_results, n_classes, hbar):
    fidelity_matrix = np.zeros((n_classes, n_classes))

    #use one distinct, correctly predicted sample per class
    representative_indices = {}

    #convert lists to numpy arrays to be safe
    all_labels = np.array(test_results['labels'])

    for c in range(n_classes):
        # find all indices where this class was true label
        class_indices = np.where(all_labels == c)[0]
        if len(class_indices) > 0:
            #use the very first occurrence of this class
            representative_indices[c] = class_indices[len(class_indices) // 2]

        else:
            representative_indices[c] = None

    for i in range(n_classes):
        idx_i = representative_indices[i]
        if idx_i is None: continue

        for j in range(n_classes):
            idx_j = representative_indices[j]
            if idx_j is None: continue

            #extract the 8-element mu and 8x8 cov
            mu_i, cov_i = test_results["mus"][idx_i], test_results["covs"][idx_i]
            mu_j, cov_j = test_results["mus"][idx_j], test_results["covs"][idx_j]

            fidelity_matrix[i, j] = compute_gaussian_fidelity(mu_i, cov_i, mu_j, cov_j, hbar)

    return fidelity_matrix

# def analyze_state_separation(test_results, n_classes, hbar):
#     # test results should contain predicted_mu, predicted_cov, true_label

#     # compute class centroids (average mu and cov per class)
#     class_centroids = {}
#     for c in range(n_classes):
#         indices = [i for i, label in enumerate(test_results['labels']) if label == c]
#         class_centroids[c] = {
#             'mu': np.mean([test_results['mus'][i] for i in indices], axis=0),
#             'cov': np.mean([test_results['covs'][i] for i in indices], axis=0)
#         }

    
#     # build the fidelity matrix
#     fidelity_matrix = np.zeros((n_classes, n_classes))
#     for i in range(n_classes):
#         for j in range(n_classes):
#             fidelity_matrix[i, j] = compute_gaussian_fidelity(
#                 class_centroids[i]['mu'], class_centroids[i]['cov'],
#                 class_centroids[j]['mu'], class_centroids[j]['cov'], hbar
#             )

#     return fidelity_matrix


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
        # "Two_mode_gates": (n_modes - 1) * depth
        "Two_mode_gates": (n_modes) * depth
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