import argparse
import torch
import datetime
import os
import numpy as np
from sklearn.decomposition import PCA

from qcore.ansatz.cv_spine_ansatz import GaussianVariationalAnsatz
from experiments.models.cv_spine_model import SpineCVQNN

from qcore.data.spine_dataset import SpineCascadeDataset

def main():
    parser = argparse.ArgumentParser(description="QStrata Spine Detection")
    parser.add_argument("--name", type=str, default="cv_spine_detector", help="Model name")
    parser.add_argument("--modes", type=int, default=4, help="qumodes")
    parser.add_argument("--depth", type=int, default=1, help="circuit depth")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--squeeze", type=float, default=0.575)
    parser.add_argument("--encoding", type=float, default=2.5)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--readout", type=str, default="dual_homodyne", choices=["homodyne", "dual_homodyne"])
    args = parser.parse_args()

    #computing hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Active processing infrastructure {device}")

    #build timestamp log structure directories
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"results/vindr_spine_detection/{args.name}_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    print(f"Session workspace log space deployed at: {run_dir}")

    #instantiate data loader
    dataset = SpineCascadeDataset(csv_file=CSV_FILE, img_dir=IMG_DIR)

    #dynamically extract total categories parsed from file setup
    total_classes = dataset.get_total_classes()
    print(f"dynamically discovered unique target")

    print("\nExtracting localized patches to construct pca feature projection matrix")
    sample_vectors = []
    for i in range(min(150, len(dataset))):
        v, _ = dataset[i]
        if torch.norm(v) > 0:
            sample_vectors.append(v.numpy())
    sample_vectors = np.stack(sample_vectors)


    pca = PCA(n_components=args.modes)
    pca.fit(sample_vectors)
    print(f"PCA finalized. Captured variance {np.sum(pca.explained_variance_ratio_):.4f}")

    data_loader = DataLoader(dataset, batch_size=args.batch, shuffle=True)

    #instantiate classes using parameters
    ansatz = GaussianVariationalAnsatz(n_modes=args.modes, depth=args.depth, squeezing_cap=args.squeeze)
    model = SpineCVQNN(ansatz=ansatz, n_classes = total_classes, encoding_multiplier=args.encoding).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    