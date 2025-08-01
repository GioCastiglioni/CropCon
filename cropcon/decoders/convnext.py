import torch.nn.functional as F
import torch
from torch import nn, Tensor

from copy import deepcopy

from typing import List
from cropcon.decoders.base import Decoder
from cropcon.encoders.base import Encoder
from cropcon.decoders.ltae import LTAE2d


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
        assert self.finetune  # the UNet encoder should always be trained

        self.model_name = 'ConvNext_Segmentation'
        self.align_corners = False
        self.topology = encoder.topology
        self.in_channels = deepcopy(self.topology)
        self.dec_topology = dec_topology
        self.multi_temporal = multi_temporal

        self.encoder = encoder

        self.pad_value=0

        self.up_blocks = ConvNextDecoder(
            encoder_channels=self.in_channels[::-1],
            decoder_channels=self.dec_topology)
        
        self.tmap = LTAE2d(
            in_channels=self.topology[-1],
            d_model=256,
            n_head=16,
            mlp=[256, self.topology[-1]],
            return_att=True,
            d_k=4,
        )
        self.temporal_aggregator = Temporal_Aggregator(mode="att_group")
        self.out_conv = nn.Conv2d(self.dec_topology[0], self.num_classes, kernel_size=1)

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
        features = [out] + skips[:-1]  # exclude stem
        stem = skips[-1] 
        out = self.up_blocks(features, stem=stem)

        return out


class ConvNextDecoder(nn.Module):
    def __init__(self, encoder_channels: List[int], decoder_channels: List[int]):
        super().__init__()
        assert len(encoder_channels) == len(decoder_channels), \
            "encoder_channels and decoder_channels must have the same length"

        self.blocks = nn.ModuleList()

        self.blocks.append(
            ConvNextStyleDecoderBlock(
                in_channels=encoder_channels[0],
                out_channels=decoder_channels[-1]
            ))

        for i in range(1, len(decoder_channels)):
            in_channels = decoder_channels[-i]  # from previous decoder output
            skip_channels = encoder_channels[i]
            out_channels = decoder_channels[-(i+1)]
            self.blocks.append(
                ConvNextStyleDecoderBlock(
                    in_channels=in_channels + skip_channels,
                    out_channels=out_channels
                ))
        
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(decoder_channels[0] + encoder_channels[-1], decoder_channels[0], kernel_size=3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(decoder_channels[0] + encoder_channels[-1], decoder_channels[0], kernel_size=3, padding=1),
            nn.GELU())

    def forward(self, features: List[torch.Tensor], stem: torch.Tensor) -> torch.Tensor:
        x = features[0]
        skips = features[1:]
        x = self.blocks[0](x)

        for i, block in enumerate(self.blocks[1:]):
            x = F.interpolate(x, size=skips[i].shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skips[i]], dim=1)
            x = block(x)

        x = F.interpolate(x, size=stem.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, stem], dim=1)
        x = self.final_upsample(x)

        return x


class LayerScaler(nn.Module):
    def __init__(self, init_value: float, dimensions: int):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones((dimensions)), 
                                    requires_grad=True)
        
    def forward(self, x):
        return self.gamma[None,...,None,None] * x


class ConvNextStyleDecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)



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