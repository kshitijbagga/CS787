# models/diffusion/style_encoder.py
import torch
import torch.nn as nn

class StyleEncoder(nn.Module):
    """
    Encodes handwriting image into conditioning vector.
    """

    def __init__(self, out_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(256, out_dim)

    def forward(self, x):
        B = x.size(0)
        h = self.conv(x).view(B, -1)
        return self.fc(h)
