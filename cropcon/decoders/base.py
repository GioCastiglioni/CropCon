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

class AttentionProjectionHead(nn.Module):
    def __init__(self, embed_dim, mlp_hidden_dim, projection_dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)

        # MLP projection head
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, projection_dim)
        )

    def forward(self, x):
        # Multihead attention (self-attention)
        attn_output, _ = self.attn(x, x, x)
        pooled = attn_output.squeeze(0)  # shape: [batch_size, embed_dim]
        # Project through MLP
        projected = self.mlp(pooled)
        # Normalize to hypersphere
        normalized = F.normalize(projected, p=2, dim=-1)
        return normalized    
