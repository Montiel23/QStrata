import matplotlib.pyplot as plt
import os
import torch
import numpy as np
import pandas as pd


import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, label_binarize

from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from scipy.stats import multivariate_normal

# global paper formatting
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 14,
    'figure.titlesize': 20
})


def diagnostic_pca_boxplot(X_pca, y_labels, run_dir, name=None):
    "X_pca np array (samples, pca_components)"
    "y_labels np array (samples,)"

    df = pd.DataFrame(X_pca, columns=[f"PCA_{i}" for i in range(X_pca.shape[1])])
    df["Class"] = y_labels.flatten()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i in range(4):
        sns.boxplot(x="Class", y=f"PCA_{i}", data=df, ax=axes[i])
        axes[i].set_title(f"Distribution of PCA component {i}")
        axes[i].grid(axis="y", linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    # plt.savefig(os.path.join(run_dir, "pca_class_separation.png"), dpi=300)
    plt.savefig(os.path.join(run_dir, f"pca_class_separation_{name}.png"), dpi=300)
    plt.close()

def plot_class_distribution(data_dict, run_dir, class_names=None):
    n_classes = data_dict["n_classes"]
    labels_train = data_dict["train"][1]
    labels_val = data_dict["val"][1]
    labels_test = data_dict["test"][1]

    #count occurrences
    train_counts = np.bincount(labels_train.flatten(), minlength=n_classes)
    val_counts = np.bincount(labels_val.flatten(), minlength=n_classes)
    test_counts = np.bincount(labels_test.flatten(), minlength=n_classes)

    x = np.arange(n_classes)
    width = 0.25

    plt.figure(figsize=(12, 6))

    plt.bar(x - width, train_counts, width, label=f"Train (n={sum(train_counts)})", color="#1f77b4")
    plt.bar(x, val_counts, width, label=f"Val (n={sum(val_counts)})", color="#ff7f0e")
    plt.bar(x + width, test_counts, width, label=f"Test (n={sum(test_counts)})", color="#2ca02c")

    plt.xlabel("Class ID")
    plt.ylabel("Number of samples")
    if class_names:
        plt.xticks(x, class_names, rotation=45, ha='right')
    else:
        plt.xticks(x)
    plt.legend()
    plt.grid(axis='y', linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "data_distribution.png"), dpi=300)

def generate_phase_diagram(results_list, run_dir):
    phase_df = pd.DataFrame(results_list)

    #foce everything to numeric
    phase_df["f1"] = pd.to_numeric(phase_df["f1"], errors="coerce")
    phase_df["noise"] = pd.to_numeric(phase_df["noise"], errors="coerce")
    phase_df["s_limit"] = pd.to_numeric(phase_df["s_limit"], errors="coerce")

    # pivot_table = phase_df.pivot(index="noise", columns="s_limit", values="f1")
    # pivot_table = phase_df.pivot(index="noise", columns="s_limit", values="f1", aggfunc="mean")
    pivot_table = phase_df.pivot_table(index="noise", columns="s_limit", values="f1", aggfunc="mean")

    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, cmap="viridis", xticklabels=True, yticklabels=True)

    plt.savefig(os.path.join(run_dir, "phase.png"), dpi=300)
    plt.close()

def plot_fidelity_matrix(f_matrix, run_dir):
    plt.figure(figsize=(10,8))
    sns.heatmap(f_matrix, fmt=".2f", annot=True, cmap='viridis', xticklabels=True, yticklabels=True)
    plt.xlabel("Class ID")
    plt.ylabel("Class ID")
    plt.savefig(os.path.join(run_dir, "fidelity.png"), dpi=300)
    plt.close()

def plot_mode_wigner(mu, cov, mode_idx, save_path):
    m = mu[2*mode_idx: 2*mode_idx+2].detach().cpu().numpy()
    v = cov[2*mode_idx: 2*mode_idx+2, 2*mode_idx: 2*mode_idx+2].detach().cpu().numpy()

    #dynamic grid: center plot on the state
    #look 5 standard deviations around the mean

    std_x = np.sqrt(v[0,0])
    std_p = np.sqrt(v[1,1])

    #expand view to always see the blob
    limit_x = max(5, abs(m[0]) + 3*std_x)
    limit_p = max(5, abs(m[1]) + 3*std_p)

    x, y = np.mgrid[-limit_x:limit_x:.1, -limit_p:limit_p:.1]
    pos = np.dstack((x,y))
    rv = multivariate_normal(m, v, allow_singular=True)



    plt.figure(figsize=(6, 5))
    cf = plt.contourf(x, y, rv.pdf(pos), cmap='viridis')
    plt.xlabel("X (Position)")
    plt.ylabel("P (Momentum)")

    cbar = plt.colorbar(cf)
    cbar.ax.tick_params()
    cbar.set_label("Wigner quasi-probability")

    plt.tight_layout()

    # plt.colorbar(label="Wigner quasi-probability")
    plt.savefig(os.path.join(save_path), dpi=300)
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
    # ax[0].set_title("Confusion Matrix")

    # 2. Multi-class ROC Curve (One-vs-Rest)
    # We binarize the output to plot a curve for each class
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # If it's binary, label_binarize returns (N, 1), we need (N, 2) for multi-label logic
    if n_classes == 2:
        RocCurveDisplay.from_predictions(
            y_true, y_probs[:, 1], ax=ax[1],
        )


        PrecisionRecallDisplay.from_predictions(
            y_true, y_probs[:, 1], ax=ax[2],
        )

        # ax[1].set_title("ROC Curve")
        # ax[2].set_title("Precision-Recall Curve")

        # Override axis labels to remove "(Positive label: 1)"
        ax[1].set_xlabel("False Positive Rate")
        ax[1].set_ylabel("True Positive Rate")
        ax[2].set_xlabel("Recall")
        ax[2].set_ylabel("Precision")


        prevalence = np.mean(y_true)
        ax[2].axhline(y=prevalence, color='r', linestyle='--', label=f'Baseline ({prevalence:.2f})')
        ax[2].legend()
        # y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))

    else:
        for i in range(n_classes):
            RocCurveDisplay.from_predictions(
                y_true_bin[:, i], 
                y_probs[:, i], 
                ax=ax[1], 
                name=f"Class {i}"
            )
        ax[1].plot([0, 1], [0, 1], "k--")
        # ax[1].set_title("Multi-class ROC")

        # 3. Multi-class Precision-Recall Curve
        for i in range(n_classes):
            PrecisionRecallDisplay.from_predictions(
                y_true_bin[:, i], 
                y_probs[:, i], 
                ax=ax[2], 
                name=f"Class {i}"
            )
        # ax[2].set_title("Multi-class Precision-Recall")

        # Override axis labels to remove "(Positive label: 1)"
        ax[1].set_xlabel("False Positive Rate")
        ax[1].set_ylabel("True Positive Rate")
        ax[2].set_xlabel("Recall")
        ax[2].set_ylabel("Precision")


        # prevalence = np.mean(y_true)
        # ax[2].axhline(y=prevalence, color='r', linestyle='--', label=f'Baseline ({prevalence:.2f})')
        ax[2].legend()


    for a in ax:
        a.tick_params(axis='both', which='major')

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
    plt.axvline(x=4, color='g', linestyle=':', label='Current Qumode Limit')
    
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
        axes[i + 1].set_title(f"n={n} (components)")
        axes[i + 1].axis('off')

    plt.tight_layout()
    # plt.savefig("results/pca_reconstruction_grid.png")
    plt.savefig(os.path.join(run_dir, "pca_reconstruction_grid.png"))
    # plt.show()
    plt.close()
    
    # Return the variance ratio for the current n_qubits for your logs
    var_retained = np.cumsum(pca_full.explained_variance_ratio_)[n_components_list[1]] # index 1 is '4'
    print(f"Analysis complete. 4 components retain {var_retained*100:.2f}% of image variance.")