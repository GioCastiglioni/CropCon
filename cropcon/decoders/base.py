import torch
import torch.nn as nn
from torch.nn import functional as F

from cropcon.encoders.base import Encoder


class Decoder(nn.Module):
    """Base class for decoders."""

    def __init__(
        self,
        encoder: Encoder,
        num_classes: int,
        finetune: bool,
    ) -> None:
        """Initialize the decoder.

        Args:
            encoder (Encoder): encoder used.
            num_classes (int): number of classes of the task.
            finetune (bool): whether the encoder is finetuned.
        """
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self.finetune = finetune

class ProjectionHead(nn.Module):
    def __init__(self, in_channels, num_layers=2, proj_channels=256):
        super().__init__()
        layers = []
        for i in range(num_layers - 1):
            layers.append(nn.Conv2d(in_channels, proj_channels, kernel_size=1))
            layers.append(nn.BatchNorm2d(proj_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = proj_channels
        layers.append(nn.Conv2d(in_channels, proj_channels, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, C, H, W]
        out = self.net(x)
        out = F.normalize(out, dim=1)  # L2 normalize across channels
        return out
