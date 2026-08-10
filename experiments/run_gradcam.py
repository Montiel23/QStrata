import argparse
import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

# Add repository root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.models.classical_resnet import ClassicalSpineResNet18
from experiments.models.cv_hqnn import HybridSpineCVModel
from experiments.models.dv_hqnn import HybridSpineDVModel
from qcore.data.npy_spine_dataset import NpySpineDataset


# =====================================================================
# 1. GRAD-CAM HOOK EXTRACTOR
# =====================================================================
class HybridGradCAM:

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.hook_a = target_layer.register_forward_hook(self.save_activation)
        self.hook_g = target_layer.register_full_backward_hook(
            self.save_gradient
        )

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_tensor, class_idx=None):
        self.model.eval()

        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1.0

        self.model.zero_grad()
        output.backward(gradient=one_hot)

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze(0)
        cam = torch.clamp(cam, min=0)

        max_val = cam.max()
        if max_val > 0:
            cam = cam / max_val

        return cam.cpu().numpy()

    def remove_hooks(self):
        self.hook_a.remove()
        self.hook_g.remove()


def overlay_heatmap(
    heatmap, original_img, alpha=0.4, colormap=cv2.COLORMAP_JET
):
    heatmap_resized = cv2.resize(
        heatmap, (original_img.shape[1], original_img.shape[0])
    )
    heatmap_color = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), colormap
    )
    overlay = cv2.addWeighted(original_img, 1 - alpha, heatmap_color, alpha, 0)
    return overlay


# =====================================================================
# 2. TARGET CONVOLUTIONAL LAYER RESOLVER
# =====================================================================
def get_target_layer(model, model_type):
    if model_type == "classical":
        return model.backbone.layer4[1].conv2
    else:
        return model.feature_extractor[7][1].conv2


# =====================================================================
# 3. MASTER GRAD-CAM EXECUTION
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Grad-CAM Interpretability Engine (Classical, DV, CV)"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["classical", "dv", "cv"],
    )
    parser.add_argument("--weights_path", type=str, required=True)
    parser.add_argument("--test_images", type=str, required=True)
    parser.add_argument("--test_labels", type=str, required=True)
    parser.add_argument("--qubits", type=int, default=4)
    parser.add_argument("--modes", type=int, default=4)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--hbar", type=float, default=2.0)
    parser.add_argument("--output_dir", type=str, default="gradcam_results")
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cpu", "cuda"]
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )
    print(f"\n[Grad-CAM] Initializing engine on: {device}")

    # Load Npy Dataset
    test_dataset = NpySpineDataset(
        images_path=args.test_images,
        labels_path=args.test_labels,
        to_spatial_rgb=True,
    )

    total_classes = test_dataset.get_total_classes()
    class_names = test_dataset.unique_classes
    print(f"[Dataset] Found {total_classes} categories: {class_names}")

    # Instantiate Model
    if args.model_type == "classical":
        model = ClassicalSpineResNet18(n_classes=total_classes).to(device)
    elif args.model_type == "dv":
        model = HybridSpineDVModel(
            n_qubits=args.qubits, depth=args.depth, n_classes=total_classes
        ).to(device)
    elif args.model_type == "cv":
        model = HybridSpineCVModel(
            n_modes=args.modes,
            depth=args.depth,
            n_classes=total_classes,
            hbar=args.hbar,
        ).to(device)

    # Load weights
    checkpoint = torch.load(
        args.weights_path, map_location=device, weights_only=False
    )
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    target_layer = get_target_layer(model, args.model_type)
    cam_extractor = HybridGradCAM(model, target_layer)

    # Scan test set and safely convert label to integer ID
    print("Scanning test set for class exemplars...")
    class_examples = {}

    for idx in range(len(test_dataset)):
        _, label_data = test_dataset[idx]

        if isinstance(label_data, torch.Tensor):
            c_idx = int(label_data.item())
        elif isinstance(label_data, (int, np.integer)):
            c_idx = int(label_data)
        else:
            label_str = str(label_data)
            if label_str in class_names:
                c_idx = class_names.index(label_str)
            else:
                continue

        if c_idx not in class_examples:
            class_examples[c_idx] = idx

        if len(class_examples) == total_classes:
            break

    # Build 4x4 Grid
    fig, axes = plt.subplots(4, 4, figsize=(18, 14))

    print("Generating Grad-CAM overlays...")

    for i in range(total_classes):
        if i not in class_examples:
            continue

        img_idx = class_examples[i]
        tensor_img, _ = test_dataset[img_idx]
        tensor_img_batch = tensor_img.unsqueeze(0).to(device)

        heatmap = cam_extractor.generate_heatmap(
            tensor_img_batch, class_idx=i
        )

        img_np = tensor_img.cpu().permute(1, 2, 0).numpy()
        img_np = (
            (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        ) * 255.0
        original_bgr = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2BGR)
        orig_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

        visual_result = overlay_heatmap(heatmap, original_bgr)
        vis_rgb = cv2.cvtColor(visual_result, cv2.COLOR_BGR2RGB)

        ax_orig = axes.flat[i * 2]
        ax_cam = axes.flat[i * 2 + 1]

        ax_orig.imshow(orig_rgb)
        ax_orig.set_title(
            f"Original: {class_names[i][:15]}\n(Class {i})",
            fontsize=10,
            fontweight="bold",
        )
        ax_orig.axis("off")

        ax_cam.imshow(vis_rgb)
        ax_cam.set_title(
            f"Grad-CAM: {class_names[i][:15]}", fontsize=10, fontweight="bold"
        )
        ax_cam.axis("off")

    plt.suptitle(
        f"Grad-CAM Class Activation Maps ({args.model_type.upper()} Model)",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()

    os.makedirs(args.output_dir, exist_ok=True)
    out_filename = os.path.join(
        args.output_dir, f"{args.model_type}_gradcam_grid.png"
    )
    plt.savefig(out_filename, dpi=300, bbox_inches="tight")
    plt.close()

    cam_extractor.remove_hooks()
    print(f"[Grad-CAM] Successfully saved grid report to: {out_filename}\n")


if __name__ == "__main__":
    main()