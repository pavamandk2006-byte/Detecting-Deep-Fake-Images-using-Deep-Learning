import torch
import torch.nn as nn
from torchvision import models


class DeepFakeDetector(nn.Module):
    """
    EfficientNet-B0 CNN for binary deepfake classification.
    Output:
        0 -> Real
        1 -> Fake
    """

    def __init__(self, pretrained=True, freeze_backbone=False):
        super().__init__()

        # Load EfficientNet-B0
        if pretrained:
            weights = models.EfficientNet_B0_Weights.DEFAULT
            self.backbone = models.efficientnet_b0(weights=weights)
        else:
            self.backbone = models.efficientnet_b0(weights=None)

        # Freeze feature extractor (optional)
        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False

        # Replace classifier
        in_features = self.backbone.classifier[1].in_features

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.backbone(x)