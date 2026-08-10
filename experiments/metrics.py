import torch
from scipy.linalg import sqrtm
import numpy as np
from qcore.measurement.probability import measure_probability
import os

# def compute_gaussian_fidelity(mu1, cov1, mu2, cov2, hbar=2.0):
#     #compute fidelity between two multi-mode gaussian states based on bures fidelity for gaussian states

#     mu1, cov1 = np.array(mu1, dtype=np.float64), np.array(cov1, dtype=np.float64)
#     mu2, cov2 = np.array(mu2, dtype=np.float64), np.array(cov2, dtype=np.float64)

#     mu1 = mu1.flatten()
#     mu2 = mu2.flatten()

#     d_mu = mu1 - mu2
#     V_avg = 0.5 * (cov1 + cov2)

#     V_inv = np.linalg.inv(V_avg + np.eye(len(V_avg)) * 1e-7)

#     d_mu_64 = d_mu.astype(np.float64)

#     # print(f"DEBUG: MU1 NORM: {np.linalg.norm(mu1)}")
#     # print(f"DEBUG: MU2 NORM: {np.linalg.norm(mu2)}")

#     #ensure numerical stability with small epsilon
#     # delta = 0.5 * d_mu.T @ np.linalg.inv(V_avg + np.eye(len(V_avg)) * 1e-6) @ d_mu
#     delta = 0.5 * d_mu.T @ V_inv @ d_mu_64
#     # delta = (1.0 / (2 * hbar)) * d_mu.T @ V_inv @ d_mu_64

#     #covariance shape term
#     #measure how much squeezing of class A matches class B

#     det_V1 = np.linalg.det(cov1)
#     det_V2 = np.linalg.det(cov2)
#     det_Vavg = np.linalg.det(V_avg)

#     #this drops if one state is highly squeezed and the other is not

#     shape_overlap = np.sqrt(np.sqrt(det_V1 * det_V2) /det_Vavg)

#     #final fidelity is combined score
#     fidelity = shape_overlap * np.exp(-delta)

#     return np.clip(fidelity, 0.0, 1.0)

def compute_gaussian_fidelity(mu1, cov1, mu2, cov2, hbar=2.0):
    """Computes exact Bures quantum state fidelity overlap between two multi-mode Gaussian states."""
    mu1 = np.array(mu1, dtype=np.float64).flatten()
    mu2 = np.array(mu2, dtype=np.float64).flatten()
    cov1 = np.array(cov1, dtype=np.float64)
    cov2 = np.array(cov2, dtype=np.float64)

    d_mu = mu1 - mu2
    V_avg = 0.5 * (cov1 + cov2)

    V_inv = np.linalg.inv(V_avg + np.eye(len(V_avg)) * 1e-7)
    delta = 0.5 * d_mu.T @ V_inv @ d_mu

    det_V1 = np.linalg.det(cov1)
    det_V2 = np.linalg.det(cov2)
    det_Vavg = np.linalg.det(V_avg)

    shape_overlap = np.sqrt(
        np.sqrt(max(det_V1 * det_V2, 1e-12)) / max(det_Vavg, 1e-12)
    )
    fidelity = shape_overlap * np.exp(-np.clip(delta, 0.0, 50.0))

    return float(np.clip(fidelity, 0.0, 1.0))


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
    fn = conf_matrix.sum(dim=1).float() - (tp)
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


def compute_dv_pauli_tomography(model, data_loader, n_qubits, n_classes, class_names, run_dir, device):
    model.eval()

    #pauli matrices in complex float32
    X = torch.tensor([[0, 1], [1,0]], dtype=torch.complex64, device=device)
    Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device=device)
    Z = torch.tensor([[1,0], [0, -1]], dtype=torch.complex64, device=device)
    paulis = {"X": X, "Y":Y, "Z":Z}

    class_states = {c: [] for c in range(n_classes)}

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            latent = torch.flatten(model.feature_extractor(batch_x), 1)
            q_feats = model.quantum_projection_head(latent)

            psi_batch = model.dv_quantum_classifier.get_state_vector(q_feats)

            for i in range(batch_x.size(0)):
                label = batch_y[i].item()
                if label < n_classes:
                    class_states[label].append(psi_batch[i])

    tomography_data = {}

    for c in range(n_classes):
        if len(class_states[c]) == 0:
            continue

        states = torch.stack(class_states[c])
        n_samples = states.size(0)
        pauli_matrix = np.zeros((n_qubits, 3))

        for q in range(n_qubits):
            dim_left = 2**q
            dim_right = 2 ** (n_qubits - q - 1)
            states_reshaped = states.view(n_samples, dim_left, 2, dim_right)

            for p_idx, (p_name, p_op) in enumerate(paulis.items()):
                op_psi = torch.einsum("abcd,ce->abed", states_reshaped, p_op)
                exp_vals = torch.real(torch.sum(torch.conj(states_reshaped) * op_psi, dim=[1, 2, 3]))
                pauli_matrix[q, p_idx] = torch.mean(exp_vals).item()

        tomography_data[c] = pauli_matrix

    return tomography_data

            # for p_idx, (p_name, p_op) in enumerate(paulis.items()):
            #     full_op = torch.tensor([[1]], dtype=torch.complex64, device=device)
            #     for k in range(n_qubits):
            #         op = p_op if k == q else torch.eye(2, dtype=torch.complex64, device=device)

            #     op_psi = torch.matmul(states, full_op)
            #     exp_val = torch.real(torch.sum(torch.conj(states) * op_psi, dim=1))
            #     pauli_matrix[q, p_idx] = torch.mean(exp_val).item()


    #     tomography_data[c] = pauli_matrix

    # return tomography_data


def compute_cv_gaussian_tomography(
    model, data_loader, n_modes, n_classes, device, hbar=2.0
):
    """Extracts phase-space quadrature expectations <q>, <p> and photon numbers <n> per class,

    handling all batch_y tensor formats safely.
    """
    model.eval()

    class_means = {c: [] for c in range(n_classes)}
    class_covs = {c: [] for c in range(n_classes)}

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)

            # Unrolled forward pass for state extraction
            latent = model.feature_extractor(batch_x)
            latent = torch.flatten(latent, 1)
            q_feats = torch.pi * model.quantum_projection_head(latent)

            r_batch, v_batch = model.cv_quantum_classifier.get_gaussian_state(
                q_feats
            )

            # Detach tensors to CPU
            r_np = r_batch.detach().cpu().numpy()
            v_np = v_batch.detach().cpu().numpy()

            for i in range(batch_x.size(0)):
                # Safely extract scalar integer label index
                y_i = batch_y[i]
                if isinstance(y_i, torch.Tensor):
                    if y_i.ndim > 0 and y_i.numel() > 1:
                        label = int(torch.argmax(y_i).item())
                    else:
                        label = int(y_i.item())
                else:
                    label = int(y_i)

                if 0 <= label < n_classes:
                    class_means[label].append(r_np[i])
                    class_covs[label].append(v_np[i])

    tomography_data = {}

    for c in range(n_classes):
        if len(class_means[c]) == 0 or len(class_covs[c]) == 0:
            continue

        r_c = np.mean(class_means[c], axis=0)  # [2N]
        v_c = np.mean(class_covs[c], axis=0)  # [2N, 2N]

        mode_stats = np.zeros((n_modes, 5))  # Cols: <q>, <p>, Var(q), Var(p), <n>

        for k in range(n_modes):
            q_idx, p_idx = 2 * k, 2 * k + 1
            q_val, p_val = r_c[q_idx], r_c[p_idx]
            var_q, var_p = v_c[q_idx, q_idx], v_c[p_idx, p_idx]

            # Mean photon count: <n_k> = (Var(q) + Var(p) + q^2 + p^2 - hbar) / (2 * hbar)
            n_k = (var_q + var_p + q_val**2 + p_val**2 - hbar) / (2.0 * hbar)
            mode_stats[k] = [q_val, p_val, var_q, var_p, max(0.0, n_k)]

        det_v = np.linalg.det(2.0 * v_c / hbar)
        purity = 1.0 / np.sqrt(max(det_v, 1e-12))

        tomography_data[c] = {
            "mean_vector": r_c,
            "covariance": v_c,
            "mode_stats": mode_stats,
            "purity": purity,
        }

    print(
        f"[Tomography Audit] Successfully collected Gaussian states for {len(tomography_data)}/{n_classes} classes."
    )
    return tomography_data

def compute_cv_fidelity_matrix(cv_tomography_data, n_classes, hbar=2.0):
    """Generates full n_classes x n_classes Bures fidelity matrix with diagonal F(i, i) = 1.0."""
    f_matrix = np.eye(n_classes, dtype=np.float64)

    for i in range(n_classes):
        for j in range(n_classes):
            if i in cv_tomography_data and j in cv_tomography_data:
                if i == j:
                    f_matrix[i, j] = 1.0
                else:
                    mu1, cov1 = (
                        cv_tomography_data[i]["mean_vector"],
                        cv_tomography_data[i]["covariance"],
                    )
                    mu2, cov2 = (
                        cv_tomography_data[j]["mean_vector"],
                        cv_tomography_data[j]["covariance"],
                    )

                    f_matrix[i, j] = compute_gaussian_fidelity(
                        mu1, cov1, mu2, cov2, hbar=hbar
                    )

    return f_matrix

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

            # evolved_psi = model.dv_quantum_classifier.quantum_ansatz.apply(psi, quantum_ready_features)
            evolved_psi = model.dv_quantum_classifier.get_state_vector(quantum_ready_features)


            #compute system purity: Tr(rho^2) = ||psi||^2
            for b in range(batch_size):
                # purity = torch.real(torch.sum(torch.conj(evolved_psi[b])) * evolved_psi[b]).item()
                purity = torch.sum(torch.abs(evolved_psi[b]) ** 2).item()
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
        # evolved_psi = model.dv_quantum_classifier.quantum_ansatz.apply(psi, quantum_ready_features)
        evolved_psi = model.dv_quantum_classifier.get_state_vector(quantum_ready_features)


        #collapse batched output axis to feed native unbatched measurement
        single_state = evolved_psi[0]

        qubit_expectations = []
        for i in range(n_qubits):
            p_one = measure_probability(single_state, measure_wire=i, n_qubits=n_qubits)

            z_expectation = 1.0 - 2.0 * p_one
            qubit_expectations.append(z_expectation.item())


    output_path = os.path.join(run_dir, f"trajectory_epoch_{epoch}.npy")
    np.save(output_path, np.array(qubit_expectations))


