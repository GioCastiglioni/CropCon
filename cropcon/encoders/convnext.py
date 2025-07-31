from collections import OrderedDict
from logging import Logger
from typing import Sequence, List

import torch
from torch import nn, Tensor
from torchvision.ops import StochasticDepth

from cropcon.encoders.base import Encoder


class ConvNext(Encoder):
    """
    Multi Temporal UTAE Encoder for Supervised Baseline, to be trained from scratch.
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

        self.in_channels = len(input_bands["optical"])  # number of optical bands
        self.topology = topology
        self.depths=[3,3,9,3]
        self.stem_features=self.topology[0]
        self.stem = ConvNextStem(self.in_channels, self.stem_features)

        in_out_widths = list(zip(topology, topology[1:]))
        # create drop paths probabilities (one for each stage)
        drop_probs = [x.item() for x in torch.linspace(0, .0, sum(self.depths))] 
        
        self.stages = nn.ModuleList(
            [ConvNexStage(self.stem_features, self.topology[0], self.depths[0], drop_p=drop_probs[0]),
                *[ConvNexStage(in_features, out_features, depth, drop_p=drop_p)
                    for (in_features, out_features), depth, drop_p in zip(
                        in_out_widths, self.depths[1:], drop_probs[1:]
                    )]])

    def forward(self, x):
        x = self.stem(x)
        out=[x]
        for stage in self.stages:
            x = stage(x)
            out.append(x)
        return out

    def load_encoder_weights(self, logger: Logger, from_scratch: bool = True) -> None:
        pass


class LayerScaler(nn.Module):
    def __init__(self, init_value: float, dimensions: int):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones((dimensions)), 
                                    requires_grad=True)
        
    def forward(self, x):
        return self.gamma[None,...,None,None] * x

class BottleNeckBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        expansion: int = 4,
        drop_p: float = .0,
        layer_scaler_init_value: float = 1e-6,
    ):
        super().__init__()
        expanded_features = out_features * expansion
        self.block = nn.Sequential(
            # narrow -> wide (with depth-wise and bigger kernel)
            nn.Conv2d(
                in_features, in_features, kernel_size=7, padding=3, bias=False, groups=in_features
            ),
            # GroupNorm with num_groups=1 is the same as LayerNorm but works for 2D data
            nn.GroupNorm(num_groups=1, num_channels=in_features),
            # wide -> wide 
            nn.Conv2d(in_features, expanded_features, kernel_size=1),
            nn.GELU(),
            # wide -> narrow
            nn.Conv2d(expanded_features, out_features, kernel_size=1),
        )
        self.layer_scaler = LayerScaler(layer_scaler_init_value, out_features)
        self.drop_path = StochasticDepth(drop_p, mode="batch")

        
    def forward(self, x: Tensor) -> Tensor:
        res = x
        x = self.block(x)
        x = self.layer_scaler(x)
        x = self.drop_path(x)
        x += res
        return x

class ConvNextStem(nn.Sequential):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(
            nn.Conv2d(in_features, out_features, kernel_size=4, stride=4),
            nn.BatchNorm2d(out_features)
        )

class ConvNexStage(nn.Sequential):
    def __init__(
        self, in_features: int, out_features: int, depth: int, **kwargs
    ):
        super().__init__(
            # add the downsampler
            nn.Sequential(
                nn.GroupNorm(num_groups=1, num_channels=in_features),
                nn.Conv2d(in_features, out_features, kernel_size=2, stride=2)
            ),
            *[
                BottleNeckBlock(out_features, out_features, **kwargs)
                for _ in range(depth)
            ],
        )