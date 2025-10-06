import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from skimage import measure
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


class CropAggregator(nn.Module):
    """
    Aggregates per-pixel projected features into per-connected-component vectors
    using attention pooling. Connected-component labeling (skimage) is run on CPU
    and is not differentiable. All operations that use pixel features are pure
    PyTorch, so gradients flow back into pixel features.
    """

    def __init__(self, feat_dim, attn_dim=None, ignore_index=-1):
        """
        feat_dim: dimensionality of per-pixel projected features (C_proj).
        attn_dim: dimension for Q/K/V (defaults to feat_dim).
        ignore_index: label id to ignore.
        """
        super().__init__()
        if attn_dim is None:
            attn_dim = feat_dim
        self.feat_dim = feat_dim
        self.attn_dim = attn_dim
        self.ignore_index = ignore_index

        # projections
        self.query_proj = nn.Linear(feat_dim, attn_dim, bias=True)
        self.key_proj   = nn.Linear(feat_dim, attn_dim, bias=True)
        self.value_proj = nn.Linear(feat_dim, attn_dim, bias=True)
        self.out_proj   = nn.Linear(attn_dim, feat_dim, bias=True)

    def forward(self, features, labels):
        """
        Args:
            features: [B, C_proj, H, W]  (requires_grad=True)
            labels:   [B, H, W]          (integer labels; can be CPU/GPU)
        Returns:
            obj_feats: [M, C_proj] (aggregated per-object vectors)
            obj_labels: [M] long
            num_objects_per_image: list[int]  (# objects in each image)
        """
        assert features.dim() == 4
        B, C, H, W = features.shape
        device = features.device
        dtype = features.dtype

        collected_feats = []
        collected_labels = []
        num_objects_per_image = []

        for b in range(B):
            # flatten pixel features: [H*W, C]
            feat_map = features[b].permute(1, 2, 0).reshape(-1, C)  # [HW, C]
            label_map = labels[b].detach().cpu().numpy()            # [H, W] on CPU

            # Build obj id map (connected components)
            obj_id_map = -np.ones_like(label_map, dtype=np.int32)
            next_obj_id = 0
            obj_label_list = []

            unique_classes = np.unique(label_map)
            for cls in unique_classes:
                if int(cls) == self.ignore_index:
                    continue
                mask_cls = (label_map == cls).astype(np.uint8)
                if mask_cls.sum() == 0:
                    continue
                cc = measure.label(mask_cls, connectivity=1)
                max_cc = int(cc.max())
                if max_cc == 0:
                    continue
                for comp in range(1, max_cc + 1):
                    obj_id_map[cc == comp] = next_obj_id
                    obj_label_list.append(int(cls))
                    next_obj_id += 1

            if next_obj_id == 0:
                # no objects for this image
                num_objects_per_image.append(0)
                continue

            num_objects_per_image.append(next_obj_id)

            # flattened obj ids [HW]
            obj_ids_flat = torch.from_numpy(obj_id_map.reshape(-1)).to(device=device, dtype=torch.long)  # [HW]

            # valid pixels (belong to some object)
            valid_mask = obj_ids_flat >= 0
            pix_obj_ids = obj_ids_flat[valid_mask]               # [N_valid]
            pix_feats = feat_map[valid_mask]                     # [N_valid, C]

            # mean per object (index_add)
            num_obj = next_obj_id
            obj_sums = torch.zeros((num_obj, C), device=device, dtype=dtype)
            obj_counts = torch.zeros((num_obj, 1), device=device, dtype=dtype)
            obj_sums.index_add_(0, pix_obj_ids, pix_feats)
            ones = torch.ones((pix_feats.size(0), 1), device=device, dtype=dtype)
            obj_counts.index_add_(0, pix_obj_ids, ones)
            obj_means = obj_sums / (obj_counts + 1e-6)  # [num_obj, C]

            # Attention pooling
            Q = self.query_proj(obj_means)     # [num_obj, D]
            K = self.key_proj(pix_feats)       # [N_valid, D]
            V = self.value_proj(pix_feats)     # [N_valid, D]

            Q_per_pix = Q[pix_obj_ids]         # [N_valid, D]
            attn_logits = (Q_per_pix * K).sum(dim=-1) / (self.attn_dim ** 0.5)  # [N_valid]

            # stable softmax per object: compute max per object
            num_obj_t = num_obj
            try:
                # Fast path: scatter_reduce amax (PyTorch >= 1.12/2.0)
                max_per_obj = torch.full((num_obj_t,), float("-inf"), device=device, dtype=dtype)
                max_per_obj = max_per_obj.scatter_reduce(0, pix_obj_ids, attn_logits, reduce="amax", include_self=True)
            except Exception:
                # Fallback: compute max per object with a CPU grouping (ok if num objects small)
                max_per_obj = torch.full((num_obj_t,), float("-inf"), device=device, dtype=dtype)
                pix_obj_ids_cpu = pix_obj_ids.cpu().numpy()
                attn_logits_cpu = attn_logits.cpu().numpy()
                for oid in range(num_obj_t):
                    mask_idx = np.where(pix_obj_ids_cpu == oid)[0]
                    if mask_idx.size == 0:
                        continue
                    max_per_obj[oid] = float(np.max(attn_logits_cpu[mask_idx]))
                max_per_obj = max_per_obj.to(device)

            attn_logits_stable = attn_logits - max_per_obj[pix_obj_ids]
            exp_scores = torch.exp(attn_logits_stable)  # [N_valid]

            # denom per object
            denom = torch.zeros((num_obj_t,), device=device, dtype=dtype)
            denom = denom.index_add(0, pix_obj_ids, exp_scores)  # [num_obj]
            weights = exp_scores / (denom[pix_obj_ids] + 1e-6)    # [N_valid]

            weighted_V = V * weights.unsqueeze(1)                # [N_valid, D]
            obj_V_sums = torch.zeros((num_obj_t, V.size(-1)), device=device, dtype=dtype)
            obj_V_sums.index_add_(0, pix_obj_ids, weighted_V)   # [num_obj, D]

            obj_feats = self.out_proj(obj_V_sums)                # [num_obj, C]

            collected_feats.append(obj_feats)
            collected_labels.append(torch.tensor(obj_label_list, device=device, dtype=torch.long))

        # concat per-batch results
        if len(collected_feats) == 0:
            return (torch.zeros((0, self.feat_dim), device=device, dtype=features.dtype),
                    torch.zeros((0,), device=device, dtype=torch.long),
                    num_objects_per_image)
        obj_feats = torch.cat(collected_feats, dim=0)      # [M, C]
        obj_labels = torch.cat(collected_labels, dim=0)    # [M]
        return obj_feats, obj_labels, num_objects_per_image

