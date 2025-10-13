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
            layers.append(nn.GroupNorm(proj_channels//16, proj_channels))
            layers.append(nn.ReLU(inplace=False))
            in_channels = proj_channels
        layers.append(nn.Conv2d(in_channels, proj_channels, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, C, H, W]
        out = self.net(x)
        out = F.normalize(out, dim=1)  # L2 normalize across channels
        return out


class PrototypeAttentionProjector(nn.Module):
    def __init__(self, in_channels, proj_channels=256, num_heads=4):
        super().__init__()
        assert proj_channels % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = proj_channels // num_heads

        # separate projection heads for Q, K, V (can also share weights for self-attention)
        self.q_proj = ProjectionHead(in_channels, proj_channels=proj_channels)
        self.k_proj = ProjectionHead(in_channels, proj_channels=proj_channels)
        self.v_proj = ProjectionHead(in_channels, proj_channels=proj_channels)

        self.out_proj = nn.Conv2d(proj_channels, proj_channels, kernel_size=1)

    def forward(self, x_q, x_kv=None):
        """
        x_q: [B, C, H, W]   -- query feature map
        x_kv: [B, C, H, W]  -- key/value feature map (same as x_q for self-attention)
        """
        if x_kv is None:
            x_kv = x_q  # self-attention case

        B, _, H, W = x_q.shape

        Q = self.q_proj(x_q)  # [B, D, H, W]
        K = self.k_proj(x_kv)
        V = self.v_proj(x_kv)

        # reshape to multi-head format
        Q = Q.view(B, self.num_heads, self.head_dim, H * W)
        K = K.view(B, self.num_heads, self.head_dim, H * W)
        V = V.view(B, self.num_heads, self.head_dim, H * W)

        # compute attention
        attn_scores = torch.einsum("bnch,bnck->bnhk", Q, K) / (self.head_dim ** 0.5)  # [B, nH, HW, HW]
        attn_weights = F.softmax(attn_scores, dim=-1)

        # apply attention to values
        out = torch.einsum("bnhk,bnck->bnch", attn_weights, V)  # [B, nH, C_h, HW]
        out = out.reshape(B, -1, H, W)  # merge heads

        out = self.out_proj(out)
        return out