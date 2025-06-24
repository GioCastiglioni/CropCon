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
    def __init__(self, embed_dim, mlp_hidden_dim, projection_dim, attention=False, num_heads=4):
        super().__init__()
        if attention: self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        else: self.attn = None

        # MLP projection head
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, projection_dim, bias=False)
        )

    def forward(self, x):
        if self.attn != None:
            x, _ = self.attn(x, x, x)
            x = x.squeeze(0)  # shape: [batch_size, embed_dim]
        # Project through MLP
        projected = self.mlp(x)
        # Normalize to hypersphere
        normalized = F.normalize(projected, p=2, dim=-1)
        return normalized    
