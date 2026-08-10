import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class ClassicalSpineResNet18(nn.Module):

    def __init__(self, n_classes=8):
        super().__init__()
        # Load ImageNet pre-trained ResNet-18 backbone
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Replace 1000-class ImageNet head with target n_classes linear layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, n_classes)

    def forward(self, x):
        return self.backbone(x)