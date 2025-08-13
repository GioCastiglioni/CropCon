import torch.nn.functional as F
import torch
from torch import nn, Tensor
from logging import Logger
from copy import deepcopy

from typing import List
from cropcon.decoders.base import Decoder
from cropcon.encoders.base import Encoder
from cropcon.decoders.ltae import LTAE2d
from timm.layers import DropPath


class ConvNext(Decoder):

    def __init__(
        self,
        encoder: Encoder,
        num_classes: int,
        finetune: bool,
        dec_topology: list,
        multi_temporal: int
    ):
        
        super().__init__(
            encoder=encoder,
            num_classes=num_classes,
            finetune=finetune,
        )

        self.model_name = 'ConvNext_Segmentation'
        self.align_corners = False
        self.topology = encoder.topology
        self.in_channels = deepcopy(self.topology)
        self.dec_topology = dec_topology
        self.multi_temporal = multi_temporal

        self.encoder = encoder

        self.pad_value=0

        self.depths = self.encoder.depths
        self.num_stage = len(self.depths)

        self.out_conv = nn.Conv2d(int(self.dec_topology[0]/2), self.num_classes, kernel_size=1) # final classifier conv

        self.upsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        for i in reversed(range(self.num_stage)):
            upsample_layer = nn.Sequential(
                    LayerNorm(self.dec_topology[i]*2, eps=1e-6, data_format="channels_first"),
                    nn.ConvTranspose2d(self.dec_topology[i]*2, int(self.dec_topology[i]/2), kernel_size=2, stride=2),
            )
            self.upsample_layers.append(upsample_layer)
        
        self.tmap = LTAE2d(
            in_channels=self.topology[-1],
            d_model=256,
            n_head=16,
            mlp=[256, self.topology[-1]],
            return_att=True,
            d_k=4,
        )
        self.temporal_aggregator = Temporal_Aggregator(mode="att_group")

    def forward(self, x, batch_positions=None, return_feats=True):
        feat_v = self.forward_features(x, batch_positions=batch_positions)
        out = self.out_conv(feat_v)
        if return_feats: return out, feat_v
        else: return out

    def forward_features(self, input: torch.Tensor, batch_positions=None) -> torch.Tensor:
        input = input.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        B, T, C, H, W = input.shape

        pad_mask = (
            (input == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
        )  # (B, T) pad mask

        # Get spatial feature maps from encoder
        feature_maps = self.encoder(input.reshape(B * T, C, H, W))  # List of feature maps
        feature_maps = [
            fm.reshape(B, T, -1, fm.shape[-2], fm.shape[-1]) for fm in feature_maps
        ]  # reshape: (B, T, C, H, W)
        if T > 1:
            # TEMPORAL ENCODER on the deepest feature map
            x_in = feature_maps[-1].permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
            out, att = self.tmap(
                x_in,
                batch_positions=batch_positions.to(x_in.device),
                pad_mask=pad_mask,
            )
            use_temporal_aggregation = True
        else:
            out = feature_maps[-1].squeeze(1)  # (B, C, H, W)
            att = None
            use_temporal_aggregation = False

        # Construct skip connections (deepest → shallowest)
        skips = []
        for feat in reversed(feature_maps[:-1]):
            if use_temporal_aggregation:
                skip = self.temporal_aggregator(
                    feat, pad_mask=pad_mask, attn_mask=att
                )
            else:
                skip = feat.squeeze(1)
            skips.append(skip)

        # DECODER (call once with bottleneck and skips)
        skips.reverse()
        skips.append(out)
        out = self.Decoder(out, skips)

        return out

    def Decoder(self, x, down_features):
        for i in range(self.num_stage):
            x = torch.cat([x, down_features.pop()], dim=1)
            x = self.upsample_layers[i](x)
        return x


class Temporal_Aggregator(nn.Module):
    def __init__(self, mode="mean"):
        super(Temporal_Aggregator, self).__init__()
        self.mode = mode

    def forward(self, x, pad_mask=None, attn_mask=None):
        if pad_mask is not None and pad_mask.any():
            if self.mode == "att_group":
                n_heads, b, t, h, w = attn_mask.shape
                attn = attn_mask.reshape(n_heads * b, t, h, w)

                if x.shape[-2] > w:
                    attn = nn.Upsample(
                        size=x.shape[-2:], mode="bilinear", align_corners=False
                    )(attn)
                else:
                    attn = nn.AvgPool2d(kernel_size=w // x.shape[-2])(attn)

                attn = attn.reshape(n_heads, b, t, *x.shape[-2:])
                attn = attn * (~pad_mask).float()[None, :, :, None, None]

                out = torch.stack(x.chunk(n_heads, dim=2))  # hxBxTxC/hxHxW
                out = attn[:, :, :, None, :, :] * out
                out = out.sum(dim=2)  # sum on temporal dim -> hxBxC/hxHxW
                out = torch.cat([group for group in out], dim=1)  # -> BxCxHxW
                return out
            elif self.mode == "att_mean":
                attn = attn_mask.mean(dim=0)  # average over heads -> BxTxHxW
                attn = nn.Upsample(
                    size=x.shape[-2:], mode="bilinear", align_corners=False
                )(attn)
                attn = attn * (~pad_mask).float()[:, :, None, None]
                out = (x * attn[:, :, None, :, :]).sum(dim=1)
                return out
            elif self.mode == "mean":
                out = x * (~pad_mask).float()[:, :, None, None, None]
                out = out.sum(dim=1) / (~pad_mask).sum(dim=1)[:, None, None, None]
                return out
        else:
            if self.mode == "att_group":
                n_heads, b, t, h, w = attn_mask.shape
                attn = attn_mask.reshape(n_heads * b, t, h, w)
                if x.shape[-2] > w:
                    attn = nn.Upsample(
                        size=x.shape[-2:], mode="bilinear", align_corners=False
                    )(attn)
                else:
                    attn = nn.AvgPool2d(kernel_size=w // x.shape[-2])(attn)
                attn = attn.reshape(n_heads, b, t, *x.shape[-2:])
                out = torch.stack(x.chunk(n_heads, dim=2))  # hxBxTxC/hxHxW
                out = attn[:, :, :, None, :, :] * out
                out = out.sum(dim=2)  # sum on temporal dim -> hxBxC/hxHxW
                out = torch.cat([group for group in out], dim=1)  # -> BxCxHxW
                return out
            elif self.mode == "att_mean":
                attn = attn_mask.mean(dim=0)  # average over heads -> BxTxHxW
                attn = nn.Upsample(
                    size=x.shape[-2:], mode="bilinear", align_corners=False
                )(attn)
                out = (x * attn[:, :, None, :, :]).sum(dim=1)
                return out
            elif self.mode == "mean":
                return x.mean(dim=1)


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
