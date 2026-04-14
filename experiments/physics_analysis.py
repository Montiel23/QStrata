import json
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_physics_metrics(metrics_path, run_dir):
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    grad_norms = np.array(metrics['grad_norm'])
    f1_scores = np.array(metrics['val_f1'])
    entropy = np.array(metrics['entropy'])
    epochs = np.arange(1, len(grad_norms) + 1)

    #compute variance of the gradients
    grad_var = np.var(grad_norms)

    #compute signal to noise ratio in training
    snr = np.mean(grad_norms) / (np.std(grad_norms) + 1e-6)

    print(f"---- Physics Analysis report ----")
    print(f"Gradient variance: {grad_var:.6e}")
    print(f"Training SNR: {snr:.4f}")

    if grad_var > 1e-4:
        print("No barren plateau detected. Ansatz is expressive")
    else:
        print("Warning: potential barren plateau. hilbert space may be too large for current depth")
        

    
    # gradient stability plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, grad_norms, marker='o', color='purple', label= 'Gradient Norm')
    plt.axhline(y=np.mean(grad_norms), color='r', linestyle='--', label='Mean standard')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel("||∇θ|| (Log Scale)")
    plt.legend()
    plt.grid(True, which='both', ls='-', alpha=0.5)
    plt.savefig(os.path.join(run_dir, "grad_stability.png"), dpi=300)
    plt.close()


    #entropy accuracy correlation
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(epochs, entropy, color="green", label="Entanglement Entropy")
    ax1.set_ylabel("Von Neumann Entropy", color="green")

    ax2 = ax1.twinx()
    ax2.plot(epochs, f1_scores, color="blue", label="Validation F1")
    ax2.set_ylabel("F1 score", color="blue")

    plt.savefig(os.path.join(run_dir, "entropy_accuracy.png"), dpi=300)
    plt.close()

