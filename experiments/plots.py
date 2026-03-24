import matplotlib.pyplot as plt
import os
import torch
import numpy as np

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