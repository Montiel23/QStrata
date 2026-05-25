"""
qcore/models/dv_hybrid_cnn_qnn.py

DV Hybrid CNN-QNN Model — Slice Q4

Combines the frozen C006-D040 CNN feature extractor with a DV quantum
classifier head (medical_ansatz) for binary PneumoniaMNIST classification.

Architecture:
    Input (B, 1, 28, 28)
    → CNN backbone model[:4]     [frozen]      → (B, 128)
    → nn.Linear(128, n_qubits)   [trainable]   → (B, n_qubits)
    → per-sample medical_ansatz loop            → (B, 2^n_qubits)
    → nn.Linear(2^n_qubits, 2)   [trainable]   → (B, 2)

Trainable parameters: projection (516), theta (24), readout (34) = 574 total

Q5 update: medical_ansatz now uses torch.atan(x[q]) for data reuploading,
restoring full differentiability through the encoding path. Projection output
is no longer detached — gradients flow from loss through readout → quantum
circuit → projection layer. All three trainable components (projection, theta,
readout) now receive gradient updates.

Do not modify any existing source files.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from qcore.models.cnn import build_model
from qcore.ansatz.medical_ansatz import medical_ansatz, get_ansatz_shape
from qcore.backends.base import Backend
from qcore.states.vacuum import vacuum_state


class DVHybridCNNQNN(nn.Module):
    """
    DV Hybrid CNN-QNN classifier for binary PneumoniaMNIST classification.

    The CNN backbone (C006-D040 architecture, model[:4]) is frozen.
    The projection layer, quantum theta, and readout are trainable.

    Parameters
    ----------
    cnn_config : dict
        Flat config dict for build_model() — keys: conv_channels, block_type,
        use_batchnorm, dropout, pooling, input_channels, num_classes.
    n_qubits : int
        Number of qubits in the DV quantum head. Default 4.
    depth : int
        Variational circuit depth. Default 1.
    alpha : float
        Scaling factor for medical_ansatz angle encoding. Default 0.1.
    n_classes : int
        Number of output classes. Default 2 (binary).
    """

    def __init__(
        self,
        cnn_config: dict,
        n_qubits: int = 4,
        depth: int = 1,
        alpha: float = 0.1,
        n_classes: int = 2,
    ) -> None:
        super().__init__()
        self.n_qubits  = n_qubits
        self.depth     = depth
        self.alpha     = alpha
        self.n_classes = n_classes

        # ── CNN backbone — frozen feature extractor ───────────────────────────
        # build_model() returns nn.Sequential with 6 layers for C006:
        #   [0] build_block(depthwise_sep, 1,   64)   → (B, 64,  28, 28)
        #   [1] build_block(depthwise_sep, 64, 128)   → (B, 128, 28, 28)
        #   [2] AdaptiveAvgPool2d((1, 1))              → (B, 128,  1,  1)
        #   [3] Flatten()                              → (B, 128)
        #   [4] Dropout(p)
        #   [5] Linear(128, 2)
        # We take [:4] — backbone output is (B, 128).
        full_model = build_model(cnn_config)
        self.backbone = nn.Sequential(*list(full_model.children())[:4])

        # Freeze all backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        # ── Projection layer — Linear(128, n_qubits), no activation ─────────
        # medical_ansatz handles angle scaling internally via arctan and alpha.
        # No activation is needed here (consistent with TwoDQClassifier pattern).
        self.proj = nn.Linear(128, n_qubits)

        # ── Quantum variational parameters ────────────────────────────────────
        # theta shape: get_ansatz_shape(4, 1) = (1, 2, 4, 3) = 24 values
        self.theta = nn.Parameter(alpha * torch.randn(*get_ansatz_shape(n_qubits, depth)))

        # Backend for circuit compilation and execution
        self._backend = Backend()

        # ── Readout layer — Linear(2^n_qubits, n_classes) ────────────────────
        # Consistent with TwoDQClassifier.readout_layer design.
        # Maps full probability distribution (16 values) to class logits.
        self.readout = nn.Linear(2 ** n_qubits, n_classes)

        # Internal: probability sums collected during forward (for monitoring)
        self._prob_sums: list[float] = []

    # ── Module mode management ─────────────────────────────────────────────────
    def train(self, mode: bool = True) -> "DVHybridCNNQNN":
        """
        Override to keep backbone always in eval mode.
        Ensures BatchNorm running stats are never updated by the frozen backbone.
        """
        super().train(mode)
        self.backbone.eval()
        return self

    # ── Parameter accounting ──────────────────────────────────────────────────
    def frozen_param_count(self) -> int:
        """Number of frozen backbone parameters."""
        return sum(p.numel() for p in self.backbone.parameters())

    def trainable_param_count(self) -> int:
        """Number of parameters that receive gradient updates."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ── Quantum forward (single sample) ──────────────────────────────────────
    def _quantum_forward_single(self, x_i: torch.Tensor) -> torch.Tensor:
        """
        Run one sample through the DV quantum circuit.

        Q5: x_i may carry requires_grad=True. medical_ansatz now uses
        torch.atan(x_i[q]) which preserves the PyTorch autograd graph,
        enabling gradient flow from the circuit output back to x_i and
        from there to the projection layer weights.

        Parameters
        ----------
        x_i : (n_qubits,) tensor — may have requires_grad=True

        Returns
        -------
        probs : (2^n_qubits,) real tensor — probability distribution over basis states
        """
        circuit = medical_ansatz(x_i, self.theta, self.n_qubits, self.depth, self.alpha)
        U     = self._backend.compile(circuit)
        state = vacuum_state(self.n_qubits)
        out   = self._backend.run(U, state)
        return torch.abs(out) ** 2  # (2^n_qubits,)

    # ── Main forward ──────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : (B, 1, 28, 28) image batch

        Returns
        -------
        logits : (B, 2) class logits
        """
        B = x.shape[0]

        # ── CNN backbone (frozen, no gradient tracking needed) ────────────────
        with torch.no_grad():
            features = self.backbone(x)        # (B, 128), requires_grad=False

        # ── Projection ────────────────────────────────────────────────────────
        proj_out = self.proj(features)         # (B, n_qubits), requires_grad=True

        # ── Per-sample quantum loop ───────────────────────────────────────────
        # Q5: proj_out[i] is passed directly — no detach. medical_ansatz now
        # uses torch.atan(x[q]) which preserves the autograd graph, enabling
        # gradient flow from loss → readout → probs → quantum circuit →
        # proj_out → proj.weight/bias (full end-to-end gradient path).
        probs_list: list[torch.Tensor] = []
        self._prob_sums = []

        for i in range(B):
            x_i    = proj_out[i]                          # (n_qubits,) — grad enabled (Q5)
            probs_i = self._quantum_forward_single(x_i)   # (2^n_qubits,)
            probs_list.append(probs_i)
            self._prob_sums.append(probs_i.sum().item())

        probs  = torch.stack(probs_list)       # (B, 2^n_qubits)

        # ── Readout ───────────────────────────────────────────────────────────
        logits = self.readout(probs)           # (B, 2)
        return logits
