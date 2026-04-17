import matplotlib.pyplot as plt
import os
import torch
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, label_binarize

from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from scipy.stats import multivariate_normal

def plot_mode_wigner(mu, cov, mode_idx, run_dir):
    #extract 2d mean and 2x2 cov for specific mode
    m = mu[2*mode_idx : 2*mode_idx+2].detach().numpy()
    v = cov[2*mode_idx : 2*mode_idx+2, 2*mode_idx : 2*mode_idx+2].detach().numpy()

    #create grid
    x, y = np.mgrid[-5:5:.05, -5:5:0.05]
    pos = np.dstack((x, y))
    rv = multivariate_normal(m, v)

    plt.figure(figsize=(6, 5))
    plt.contourf(x, y, rv.pdf(pos), cmap='viridis')
    plt.xlabel("X (Position)")
    plt.ylabel("P (Momentum)")
    plt.colorbar(label="Wigner quasi-probability")
    plt.savefig(os.path.join(run_dir, "wigner_function.png"), dpi=300)
    plt.close()


    
def plot_curves(values, name, run_dir):
    plt.figure()
    plt.plot(values)
    plt.xlabel("Epoch")
    plt.ylabel(name)
    plt.savefig(os.path.join(run_dir, f"{name}.png"), dpi=300)
    plt.close()


def plot_boundary(model, X, y, name, run_dir, resolution=60):

    x_min, x_max = X[:, 0].min()-0.5, X[:, 0].max()+0.5
    y_min, y_max = X[:, 1].min()-0.5, X[:, 1].max()+0.5

    xs = np.linspace(x_min, x_max, resolution)
    ys = np.linspace(y_min, y_max, resolution)

    grid = np.zeros((resolution, resolution))

    with torch.no_grad():
        for i, x in enumerate(xs):
            for j, y_ in enumerate(ys):

                point = torch.tensor([x, y_], dtype=torch.float32)

                p, _ = model.forward(point)

                grid[j, i] = p.item()

        plt.figure(figsize=(6, 6))

        plt.contourf(xs, ys, grid, levels=50)

        plt.scatter(
            X[:, 0],
            X[:, 1],
            c=y,
            # edgecolors="black"
        )

        plt.savefig(os.path.join(run_dir, f"{name}.png"), dpi=300)
        plt.close()


def plot_inference_report_multiclass(y_true, y_logits, run_dir, n_classes):
    fig, ax = plt.subplots(1, 3, figsize=(20, 6))

    y_true = np.array(y_true)
    # Convert logits to probabilities using Softmax for the curves
    y_probs = torch.softmax(torch.tensor(y_logits), dim=1).numpy()
    y_preds = np.argmax(y_probs, axis=1)

    # 1. Multi-class Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_preds, ax=ax[0], cmap="Blues", colorbar=False
    )
    ax[0].set_title("Confusion Matrix")

    # 2. Multi-class ROC Curve (One-vs-Rest)
    # We binarize the output to plot a curve for each class
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # If it's binary, label_binarize returns (N, 1), we need (N, 2) for multi-label logic
    if n_classes == 2:
        y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))

    for i in range(n_classes):
        RocCurveDisplay.from_predictions(
            y_true_bin[:, i], 
            y_probs[:, i], 
            ax=ax[1], 
            name=f"Class {i}"
        )
    ax[1].plot([0, 1], [0, 1], "k--")
    ax[1].set_title("Multi-class ROC")

    # 3. Multi-class Precision-Recall Curve
    for i in range(n_classes):
        PrecisionRecallDisplay.from_predictions(
            y_true_bin[:, i], 
            y_probs[:, i], 
            ax=ax[2], 
            name=f"Class {i}"
        )
    ax[2].set_title("Multi-class Precision-Recall")

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "inference_report.png"), dpi=300)
    plt.close()
    

def plot_inference_report(y_true, y_probs, run_dir):
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    y_true = np.array(y_true)
    y_probs = np.array(y_probs)

    #confusion matrix
    y_preds = (y_probs > 0.5).astype(int)



    ConfusionMatrixDisplay.from_predictions(y_true, y_preds, ax=ax[0], cmap="Blues")
    # ax[0].set_title("Confusion Matrix")

    #roc curve
    RocCurveDisplay.from_predictions(y_true, y_probs, ax=ax[1])
    ax[1].plot([0, 1], [0, 1], "k--")
    
    #precision recall curve
    PrecisionRecallDisplay.from_predictions(y_true, y_probs, ax=ax[2])

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "inference_report.png"), dpi=300)
    plt.close()


def analyze_pca(data, n_components_list=[2, 4, 8, 16, 32], run_dir="results"):
    """
    Analyzes how much medical information is lost across different quantum bottlenecks.
    Args:
        data: The dictionary returned by load_robust_medmnist
    """
    # 1. Get the raw, original images we saved in the loader
    original_imgs = data['original_images'] 
    n_samples = len(original_imgs)
    
    # 2. Prepare the data for a "Full" PCA analysis
    X_flat = original_imgs.reshape(n_samples, -1).astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat)

    # --- PART A: Variance Analysis (Scree Plot) ---
    # We fit a PCA with the maximum possible components to see the full curve
    pca_full = PCA().fit(X_scaled)
    
    plt.figure(figsize=(10, 5))
    plt.plot(np.cumsum(pca_full.explained_variance_ratio_), 'd-', color='#2c3e50')
    plt.axhline(y=0.9, color='r', linestyle='--', label='90% Information')
    
    # Highlight your current n_qubits (assuming it's the first in the list or usually 4)
    plt.axvline(x=4, color='g', linestyle=':', label='Current Qubit Limit')
    
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Medical Information Retention Analysis")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs("results", exist_ok=True)
    # plt.savefig("results/pca_variance_scree.png")
    plt.savefig(os.path.join(run_dir, "pca_variance_scree.png"))
    plt.close()

    # --- PART B: Visual Reconstruction Grid ---
    fig, axes = plt.subplots(1, len(n_components_list) + 1, figsize=(22, 5))
    
    # Show the actual original image
    axes[0].imshow(original_imgs[0], cmap='gray')
    axes[0].set_title("Original (784 px)")
    axes[0].axis('off')

    for i, n in enumerate(n_components_list):
        # We must create a NEW PCA for each 'n' to see the difference
        pca_temp = PCA(n_components=n)
        X_pca_temp = pca_temp.fit_transform(X_scaled)
        
        # Project back to pixel space
        X_recon = pca_temp.inverse_transform(X_pca_temp)
        X_recon = scaler.inverse_transform(X_recon)
        
        # Display the "Quantum-eye view"
        axes[i + 1].imshow(X_recon[0].reshape(28, 28), cmap='gray', vmin=0, vmax=1)
        axes[i + 1].set_title(f"n={n} (Qubits)")
        axes[i + 1].axis('off')

    plt.tight_layout()
    # plt.savefig("results/pca_reconstruction_grid.png")
    plt.savefig(os.path.join(run_dir, "pca_reconstruction_grid.png"))
    # plt.show()
    plt.close()
    
    # Return the variance ratio for the current n_qubits for your logs
    var_retained = np.cumsum(pca_full.explained_variance_ratio_)[n_components_list[1]] # index 1 is '4'
    print(f"Analysis complete. 4 Qubits retain {var_retained*100:.2f}% of image variance.")