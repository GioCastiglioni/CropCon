# This code was extracted and adapted from the original implementation
# https://github.com/nasaharvest/galileo

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from einops import rearrange, repeat
from .base import Encoder as BaseEncoder
import collections
import itertools

S1_BANDS = ["VV", "VH"]
S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
SPACE_TIME_BANDS = S1_BANDS + S2_BANDS + ["NDVI"]

SPACE_TIME_BANDS_GROUPS_IDX = OrderedDict(
    {
        "S1": [SPACE_TIME_BANDS.index(b) for b in S1_BANDS],
        "S2_RGB": [SPACE_TIME_BANDS.index(b) for b in ["B2", "B3", "B4"]],
        "S2_Red_Edge": [SPACE_TIME_BANDS.index(b) for b in ["B5", "B6", "B7"]],
        "S2_NIR_10m": [SPACE_TIME_BANDS.index(b) for b in ["B8"]],
        "S2_NIR_20m": [SPACE_TIME_BANDS.index(b) for b in ["B8A"]],
        "S2_SWIR": [SPACE_TIME_BANDS.index(b) for b in ["B11", "B12"]],
        "NDVI": [SPACE_TIME_BANDS.index("NDVI")],
    }
)

def get_1d_sincos_pos_embed_from_grid_torch(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, device=pos.device) / embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = torch.einsum("m,d->md", pos, omega)
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    emb = torch.cat([emb_sin, emb_cos], dim=1)
    return emb

def get_month_encoding_table(embed_dim):
    assert embed_dim % 2 == 0
    angles = torch.arange(0, 13) / (12 / (2 * np.pi))
    sin_table = torch.sin(torch.stack([angles for _ in range(embed_dim // 2)], axis=-1))
    cos_table = torch.cos(torch.stack([angles for _ in range(embed_dim // 2)], axis=-1))
    month_table = torch.concatenate([sin_table[:-1], cos_table[:-1]], axis=-1)
    return month_table

def get_2d_sincos_pos_embed_with_resolution(embed_dim, grid_size, res, device="cpu"):
    res = res.to(device)
    grid_h = torch.arange(grid_size, device=device)
    grid_w = torch.arange(grid_size, device=device)
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")
    grid = torch.stack(grid, dim=0)
    grid = torch.einsum("chw,n->cnhw", grid, res)
    _, n, h, w = grid.shape
    
    def get_2d_sincos_pos_embed_from_grid_torch(embed_dim, grid):
        assert embed_dim % 2 == 0
        emb_h = get_1d_sincos_pos_embed_from_grid_torch(embed_dim // 2, grid[0])
        emb_w = get_1d_sincos_pos_embed_from_grid_torch(embed_dim // 2, grid[1])
        emb = torch.cat([emb_h, emb_w], dim=1)
        return emb

    pos_embed = get_2d_sincos_pos_embed_from_grid_torch(embed_dim, grid)
    pos_embed = pos_embed.reshape(n, h * w, embed_dim)
    return pos_embed

def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
        return tuple(x)
    return tuple(itertools.repeat(x, 2))

class FlexiPatchEmbed(nn.Module):
    def __init__(self, patch_size, in_chans=3, embed_dim=128):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )

    def forward(self, x, patch_size=None):
        if patch_size is None:
            patch_size = self.patch_size
        
        x = rearrange(x, "b h w t c -> (b t) c h w")
        x = self.proj(x)
        x = rearrange(x, "b c h w -> b h w c")
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class GalileoBackbone(nn.Module):
    def __init__(
        self,
        embedding_size=128,
        depth=2,
        mlp_ratio=2,
        num_heads=8,
        max_sequence_length=24,
        patch_size=16,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.patch_size = patch_size
        self.space_time_groups = SPACE_TIME_BANDS_GROUPS_IDX
        
        self.space_time_embed = nn.ModuleDict({
            group_name: FlexiPatchEmbed(in_chans=len(group), embed_dim=embedding_size, patch_size=patch_size)
            for group_name, group in self.space_time_groups.items()
        })
        
        self.space_embed = nn.ModuleDict({
            k: FlexiPatchEmbed(in_chans=len(v), embed_dim=embedding_size, patch_size=patch_size) 
            for k, v in {"SRTM": [0,1], "DW": list(range(9)), "WC": list(range(5))}.items()
        })
        
        self.time_embed = nn.ModuleDict({
             k: nn.Linear(len(v), embedding_size) for k, v in {"ERA5": [0,1], "TC": [0,1,2], "VIIRS": [0]}.items()
        })
        self.static_embed = nn.ModuleDict({
            k: nn.Linear(len(v), embedding_size) for k, v in {"LS": [0], "location": [0,1,2], "DW_static": list(range(9)), "WC_static": list(range(5))}.items()
        })

        self.pos_embed = nn.Parameter(
            get_1d_sincos_pos_embed_from_grid_torch(int(embedding_size * 0.25), torch.arange(max_sequence_length)),
            requires_grad=False,
        )
        month_tab = get_month_encoding_table(int(embedding_size * 0.25))
        self.month_embed = nn.Embedding.from_pretrained(month_tab, freeze=True)

        self.s_t_channel_embed = nn.Parameter(torch.zeros(len(self.space_time_groups), int(embedding_size * 0.25)))
        self.sp_channel_embed = nn.Parameter(torch.zeros(3, int(embedding_size * 0.25)))
        self.t_channel_embed = nn.Parameter(torch.zeros(3, int(embedding_size * 0.25))) 
        self.st_channel_embed = nn.Parameter(torch.zeros(4, int(embedding_size * 0.25)))

        # Transformer Blocks
        self.blocks = nn.ModuleList([
            Block(embedding_size, num_heads, mlp_ratio, qkv_bias=True, norm_layer=nn.LayerNorm)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embedding_size)

    def forward_features(self, s_t_x, months, input_res=10.0, output_indices=None):
        # s_t_x shape: [B, H, W, T, C_total]
        b, h, w, t, _ = s_t_x.shape
        patch_size = self.patch_size
        
        s_t_tokens = []
        for idx, (group_name, channel_idxs) in enumerate(self.space_time_groups.items()):
            group_x = s_t_x[..., channel_idxs]
            # tokens: [B*T, h', w', D]
            tokens = self.space_time_embed[group_name](group_x)
            s_t_tokens.append(tokens)
            
        # Stack groups: [B*T, h', w', G, D]
        s_t_tokens = torch.stack(s_t_tokens, dim=-2)
        
        # [B*T, h', w', G, D] -> [B, h', w', T, G, D]
        s_t_tokens = rearrange(s_t_tokens, "(b t) h w g d -> b h w t g d", b=b, t=t)
        
        new_h, new_w = s_t_tokens.shape[1], s_t_tokens.shape[2]
        dim_part = int(self.embedding_size * 0.25)
        
        # Channel Embed: [B, H, W, T, G, D_part]
        c_emb = repeat(self.s_t_channel_embed, "g d -> b h w t g d", b=b, h=new_h, w=new_w, t=t)
        
        # Pos Embed (Time): [B, H, W, T, G, D_part]
        safe_t = min(t, self.pos_embed.shape[0])
        p_emb = self.pos_embed[:safe_t]
        if t > safe_t:
             p_emb = torch.cat([p_emb, repeat(p_emb[-1:], "1 d -> k d", k=t-safe_t)], dim=0)
        p_emb = repeat(p_emb, "t d -> b h w t g d", b=b, h=new_h, w=new_w, g=len(self.space_time_groups))
        
        # Month Embed: [B, H, W, T, G, D_part]
        m_emb = self.month_embed(months) 
        m_emb = repeat(m_emb, "b t d -> b h w t g d", h=new_h, w=new_w, g=len(self.space_time_groups))
        
        # Spatial Embed: [B, H, W, T, G, D_part]
        token_res = input_res * patch_size
        gsd_ratio = token_res / 10.0 
        sp_emb = get_2d_sincos_pos_embed_with_resolution(
            dim_part, new_h, torch.ones(b, device=s_t_x.device) * gsd_ratio, device=s_t_x.device
        ) 
        sp_emb = rearrange(sp_emb, "b (h w) d -> b h w d", h=new_h, w=new_w)
        sp_emb = repeat(sp_emb, "b h w d -> b h w t g d", t=t, g=len(self.space_time_groups))
        
        total_emb = torch.cat([c_emb, p_emb, m_emb, sp_emb], dim=-1)
        
        x = s_t_tokens + total_emb
        
        x = rearrange(x, "b h w t g d -> b (h w t g) d")
        
        features = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if output_indices and i in output_indices:
                x_rec = rearrange(x, "b (h w t g) d -> b h w t g d", h=new_h, w=new_w, t=t)
                x_pool = torch.mean(x_rec, dim=(3, 4)) # Mean sobre Time y Groups
                feat = x_pool.permute(0, 3, 1, 2).contiguous() 
                features.append(feat)
             
        return features

class GalileoTiny(BaseEncoder):
    def __init__(
        self,
        encoder_weights=None,
        model_name="",
        input_size=224,
        input_bands=None,
        embed_dim=128, 
        output_layers=[1],
        output_dim=128,
        download_url=None,
        patch_size=16,
        depth=2,
        num_heads=8,
        mlp_ratio=2,
        projection_dim=64,
        positional_encoding="normal",
    ):
        super().__init__(
            model_name=model_name,
            encoder_weights=encoder_weights,
            input_bands=input_bands,
            input_size=input_size,
            embed_dim=embed_dim,
            output_layers=output_layers,
            output_dim=output_dim,
            multi_temporal=False,
            multi_temporal_output=False,
            pyramid_output=False,
            download_url=download_url,
            positional_encoding=positional_encoding
        )
        self.topology = [output_dim for _ in self.output_layers]
        
        self.patch_size = patch_size
        self.output_layers = output_layers
        
        self.backbone = GalileoBackbone(
            embedding_size=embed_dim,
            depth=depth,
            mlp_ratio=mlp_ratio,
            num_heads=num_heads,
            patch_size=patch_size,
            max_sequence_length=24 # Default
        )

        self.projector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(embed_dim, 2048),
            nn.LayerNorm(normalized_shape=2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.LayerNorm(normalized_shape=2048),
            nn.GELU(),
            nn.Linear(2048, projection_dim)
        )
        
        self.s2_indices_map = {
            "B2": 1, "B3": 2, "B4": 3, "B5": 4, "B6": 5, 
            "B7": 6, "B8": 7, "B8A": 8, "B11": 10, "B12": 11
        }

    def _prepare_input(self, x):
        
        b, t, c, h, w = x.shape
        device = x.device
        
        s1 = torch.zeros(b, t, 2, h, w, device=device)
        
        s2_bands = []
        for b_name in ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]:
            idx = self.s2_indices_map[b_name]
            s2_bands.append(x[:, :, idx:idx+1, :, :])
        s2 = torch.cat(s2_bands, dim=2)
        
        b8 = x[:, :, 7:8, :, :]
        b4 = x[:, :, 3:4, :, :]
        ndvi = (b8 - b4) / (b8 + b4 + 1e-6)
        
        x_galileo = torch.cat([s1, s2, ndvi], dim=2) # [B, T, 13, H, W]
        
        # Rearrange to expected [B, H, W, T, C]
        x_galileo = x_galileo.permute(0, 3, 4, 1, 2)
        return x_galileo

    def forward(self, x, batch_positions=None):
        """
        Args:
            x: Tensor [B, C, T, H, W] 
            batch_positions: Tensor [B, T]
        """
        # Standarize input to [B, T, C, H, W]
        x = x.permute(0, 2, 1, 3, 4)
        B, T, C, H, W = x.shape
        
        pad_mask = (x == 0).all(dim=2).all(dim=2).all(dim=2) # [B, T]
        
        s_t_x = self._prepare_input(x)
        
        if batch_positions is not None:
            if isinstance(batch_positions, dict):
                 doy = batch_positions.get("doy", torch.zeros(B, T, device=x.device))
            else:
                 doy = batch_positions
            
            months = ((doy * 365 - 1) / 30.5).long().clamp(0, 11)
        else:
            months = torch.zeros(B, T, dtype=torch.long, device=x.device)

        # Forward Backbone
        features = self.backbone.forward_features(
            s_t_x, 
            months=months, 
            input_res=10.0,
            output_indices=self.output_layers
        )
        
        out = features[-1] if features else None
        
        return out, features, pad_mask, None

    def load_encoder_weights(self, logger=None, from_scratch=False):
        if from_scratch or self.encoder_weights is None:
            return

        if logger: logger.info(f"Loading GalileoTiny weights from {self.encoder_weights}...")
        
        try:
            state_dict = torch.load(self.encoder_weights, map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "model" in state_dict:
                 state_dict = state_dict["model"]
            
            own_state = self.backbone.state_dict()
            new_state = {}
            
            for k, v in state_dict.items():
                key = k.replace("encoder.", "").replace("backbone.", "")
                
                if key in own_state:
                    target_shape = own_state[key].shape
                    
                    # 8x8 -> 16x16
                    if "proj.weight" in key and v.shape != target_shape:
                        if v.dim() == 4: # Conv2d weight [Out, In, kH, kW]
                            if logger: logger.info(f"Interpolating {key} from {v.shape} to {target_shape}")
                            v = F.interpolate(
                                v, size=(target_shape[2], target_shape[3]), 
                                mode='bicubic', align_corners=False
                            )
                    
                    if "pos_embed" in key and v.shape != target_shape:
                         # v: [T, D]
                         if logger: logger.info(f"Interpolating {key} from {v.shape} to {target_shape}")
                         # Simple linear interp for time dimension if needed
                         v = F.interpolate(
                             v.unsqueeze(0).transpose(1,2), 
                             size=(target_shape[0]), mode='linear'
                         ).transpose(1,2).squeeze(0)

                    new_state[key] = v
            
            missing, unexpected = self.backbone.load_state_dict(new_state, strict=False)
            
            if logger:
                logger.info("GalileoTiny weights loaded.")
                if missing: logger.warning(f"Missing keys: {missing}")
                
        except Exception as e:
            if logger: logger.error(f"Error loading weights: {e}")
            raise e

    def __str__(self):
        return "GalileoTiny"