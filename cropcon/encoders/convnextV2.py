from typing import Sequence
from logging import Logger

import torch
from torch import nn

from cropcon.encoders.base import Encoder

from torch.nn import functional as F
from timm.layers import DropPath

class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GRN(nn.Module):
    """Global Response Normalization for NCHW layout (fast)."""
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta  = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        # L2 norm over spatial dims via sum of squares (much faster than torch.linalg.norm)
        Gx2 = x.mul(x).sum(dim=(2, 3), keepdim=True)             # (N, C, 1, 1)
        Gx  = torch.sqrt(Gx2 + self.eps)                         # (N, C, 1, 1)
        Nx  = Gx / (Gx.mean(dim=1, keepdim=True) + self.eps)     # (N, C, 1, 1)
        return x + self.gamma * (x * Nx) + self.beta

class Block(nn.Module):
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv  = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm    = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act     = nn.GELU()
        self.grn     = GRN(4 * dim)
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.contiguous()                     # (defensive; cheap)
        x = self.norm(x)                       # channels_first LN
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)                        # fast GRN above
        x = self.pwconv2(x)
        x = shortcut + self.drop_path(x)
        return x


class ConvNext(Encoder):
    """
    Multi Temporal Convnext Encoder for Supervised Baseline, to be trained from scratch.
    It supports single time frame inputs with optical bands

    Args:
        input_bands (dict[str, list[str]]): Band names, specifically expecting the 'optical' key with a list of bands.
        input_size (int): Size of the input images (height and width).
        topology (Sequence[int]): The number of feature channels at each stage of the U-Net encoder.

    """

    def __init__(
        self,
        input_bands: dict[str, list[str]],
        input_size: int,
        multi_temporal: int,
        topology: Sequence[int],
        output_dim: int | list[int],
        download_url: str,
        encoder_weights: str | None = None,
    ):
        super().__init__(
            model_name="ConvNext",
            encoder_weights=encoder_weights,  # no pre-trained weights, train from scratch
            input_bands=input_bands,
            input_size=input_size,
            embed_dim=0,
            output_dim=output_dim,
            output_layers=None,
            multi_temporal=multi_temporal,
            multi_temporal_output=False,
            pyramid_output=True,
            download_url=download_url,
        )
        self.depths = [3, 3, 9, 3]
        self.num_stage = len(self.depths)
        self.in_channels = len(input_bands["optical"])
        self.topology = topology
        drop_path_rate = 0.

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers

        stem = nn.Sequential(
            nn.Conv2d(self.in_channels, self.topology[0], kernel_size=2, stride=2),
            LayerNorm(self.topology[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)

        for i in range(self.num_stage - 1):
            downsample_layer = nn.Sequential(
                    LayerNorm(self.topology[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(self.topology[i], self.topology[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))] 
        cur = 0
        for i in range(self.num_stage):
            stage = nn.Sequential(
                *[Block(dim=self.topology[i], drop_path=dp_rates[cur + j]) for j in range(self.depths[i])]
            )
            self.stages.append(stage)
            cur += self.depths[i]

        self.norm = nn.LayerNorm(self.topology[-1], eps=1e-6) # final norm layer

    def forward(self, x):
        down_features = []
        for i in range(self.num_stage):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            down_features.append(x)

        return down_features

    def load_encoder_weights(self, logger: Logger, from_scratch: bool = True) -> None:
        pass