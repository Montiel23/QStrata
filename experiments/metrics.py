import torch
from scipy.linalg import sqrtm
import numpy as np
from qcore.measurement.probability import measure_probability

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


def get_stats(conf_matrix):
    tp = torch.diag(conf_matrix).float()
    fp = conf_matrix.sum(dim=0).float() - tp
    fn = conf_matrix.sum(dim=1).float() - (tp + fp + fn)
    tn = conf_matrix.sum().float() - (tp + fp + fn)
    return tp, fp, tn, fn


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


def audit_quantum_register(model, data_loader, n_qubits, n_classes, device):
    model.eval()
    state_dim = 2 ** n_qubits

    class_states = {c: torch.zeros(state_dim, device=device, dtype=torch.complex64) for c in range(n_classes)}
    class_counts = {c: 0 for c in range(n_classes)}

    purities = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)

            #pass data through classical backbone and projection head
            latent = model.feature_extractor(images)
            latent = torch.flatten(latent, 1)
            quantum_ready_features = model.quantum_projection_head(latent)
            
            #regenerate state vectors using the ansatz parameters
            batch_size = images.size(0)
            psi = torch.zeros(batch_size, state_dim, device=device, dtype=torch.complex64)
            psi[:,0] = 1.0 + 0.0j

            evolved_psi = model.dv_quantum_classifier.quantum_ansatz.apply(psi, quantum_ready_features)

            #compute system purity: Tr(rho^2) = ||psi||^2
            for b in range(batch_size):
                purity = torch.real(torch.sum(torch.conj(evolved_psi[b])) * evolved_psi[b]).item()
                purities.append(purity)

                lbl = labels[b].item()

                #global phase alignment
                #extract phase angle
                global_phase = torch.angle(evolved_psi[b, 0])
                phase_rotation = torch.exp(-1j * global_phase)
                aligned_psi = evolved_psi[b] * phase_rotation

                #accumulate phase-aligned pure states safely
                class_states[lbl] += aligned_psi
                class_counts[lbl] += 1

        


    #normalize vectors to calculate centroid fidelity maps
    mean_vectors = {}
    for c in range(n_classes):
        if class_counts[c] > 0:
            norm = torch.norm(class_states[c])
            mean_vectors[c] = class_states[c] / norm if norm > 0 else class_states[c]

        else:
            mean_vectors[c] = torch.zeros(state_dim, device=device, dtype=torch.complex64)

    #compute uhlmann-jozsa state fidelity matrix
    fidelity_matrix = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            if class_counts[i] > 0 and class_counts[j] > 0:
                overlap = torch.sum(torch.conj(mean_vectors[i]) * mean_vectors[j])
                fidelity_matrix[i, j] = (torch.abs(overlap) ** 2).item()


    return fidelity_matrix, np.mean(purities)



def save_qubit_trajectory(model, sample_image, epoch, run_dir, device):
    model.eval()
    n_qubits = model.n_qubits

    with torch.no_grad():
        x = sample_image.unsqueeze(0).to(device)
        latent = model.feature_extractor(x)
        latent = torch.flatten(latent, 1)
        quantum_ready_features = model.quantum_projection_head(latent)

        #propagate state through the ansatz
        state_dim = 2 ** n_qubits
        psi = torch.zeros(1, state_dim, device=device, dtype=torch.complex64)
        psi[:, 0] = 1.0 + 0.0j
        evolved_psi = model.dv_quantum_classifier.quantum_ansatz.apply(psi, quantum_ready_features)

        #collapse batched output axis to feed native unbatched measurement
        single_state = evolved_psi[0]

        qubit_expectations = []
        for i in range(n_qubits):
            p_one = measure_probability(single_state, measure_wire=i, n_qubits=n_qubits)

            z_expectation = 1.0 - 2.0 * p_one
            qubit_expectations.append(z_expectation.item())


    output_path = os.path.join(run_dir, f"trajectory_epoch_{epoch}.npy")
    np.save(output_path, np.array(qubit_expectations))


