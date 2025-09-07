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
            x = x.unsqueeze(0)
            x, _ = self.attn(x, x, x)
            x = x.squeeze(0)  # shape: [batch_size, embed_dim]
        # Project through MLP
        projected = self.mlp(x)
        # Normalize to hypersphere
        normalized = F.normalize(projected, p=2, dim=-1)
        return normalized    


class QueryMultiHeadPoolHybrid(nn.Module):
    """
    Atención con query = mean(instance) [+ optional learned bias].
    
    Args:
        D (int): dimensión de los embeddings por pixel
        num_heads (int): número de cabezas de atención
        dropout (float): dropout en MultiheadAttention
        use_learned_bias (bool): si True, añade un vector aprendible al query
        detach_mean (bool): si True, no propaga gradiente por el promedio
    """
    def __init__(self, D, num_heads=4, dropout=0.1, use_learned_bias=True, detach_mean=True):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=D,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.use_learned_bias = use_learned_bias
        self.detach_mean = detach_mean
        if self.use_learned_bias:
            self.query_bias = nn.Parameter(torch.zeros(1, 1, D))  # (1, 1, D)

    def forward(self, x, mask=None):
        """
        x: (N, D) o (B, N, D)  -> features de píxeles
        mask: (B, N) opcional, True para píxeles válidos
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (1, N, D)
            single = True
        else:
            single = False

        B, N, D = x.shape

        # promedio de los píxeles
        mean_q = x.mean(dim=1, keepdim=True)  # (B, 1, D)
        if self.detach_mean:
            mean_q = mean_q.detach()

        # añadir bias aprendible si corresponde
        if self.use_learned_bias:
            q = mean_q + self.query_bias.expand(B, -1, -1)
        else:
            q = mean_q

        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask  # MHA espera True = padding

        out, attn_weights = self.mha(
            q, x, x,
            key_padding_mask=key_padding_mask,
            need_weights=True
        )
        out = out.squeeze(1)        # (B, D)
        attn_weights = attn_weights.squeeze(1)  # (B, N)

        if single:
            out = out.squeeze(0)        # (D,)
            attn_weights = attn_weights.squeeze(0)  # (N,)

        return out
