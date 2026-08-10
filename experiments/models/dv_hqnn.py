import torch
import torch.nn as nn
import torchvision.models as models
from qcore.ansatz.dv_spine_ansatz import DV_QNN_Spine_Classifier

class HybridSpineDVModel(nn.Module):
    def __init__(self, n_qubits, n_classes, depth=2, pretrain_backbone=True):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_classes = n_classes
        self.depth = depth

        #feature extraction with resnet18
        # base_model = models.resnet18(pretrained=pretrain_backbone)
        weights = models.ResNet18_Weights.DEFAULT if pretrain_backbone else None
        base_model = models.resnet18(weights=weights)

        #extract all layers except original fully-connected classification head
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])

        #custom reduction to compress dimensions
        self.quantum_projection_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, n_qubits),
            nn.Tanh() #bound features to [-1, 1] for optimal Bloch Sphere mapping
        )

        #dv quantum classifier
        self.dv_quantum_classifier = DV_QNN_Spine_Classifier(n_qubits, n_classes, depth=depth)

        # self.dv_quantum_classifier = nn.Linear(n_qubits, n_classes)

    def forward(self, x):
        
        # raw high-level spatial descriptors
        latent_features = self.feature_extractor(x) #shape [batch, 512, 1, 1]
        latent_features = torch.flatten(latent_features, 1) # [batch, 512]

        #contract the features to a low-dimensional 1D quantum input vector and scale [-1, 1] to [-pi, pi] for full bloch sphere coverage
        quantum_ready_vector = torch.pi *  self.quantum_projection_head(latent_features)

        # propagate the 1D feature vector through the DV rotation gates

        logits = self.dv_quantum_classifier(quantum_ready_vector)
        return logits



