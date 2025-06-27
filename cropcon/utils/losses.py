import torch
from torch.nn import functional as F


class WeightedCrossEntropy(torch.nn.Module):
    def __init__(self, ignore_index: int, distribution: list[float]) -> None:
        super(WeightedCrossEntropy, self).__init__()
        # Initialize the weights based on the given distribution
        self.weights = [1 / w if w!=0 else 0 for w in distribution]

        # Convert weights to a tensor and move to CUDA
        loss_weights = torch.Tensor(self.weights).to("cuda")
        self.loss = torch.nn.CrossEntropyLoss(
            ignore_index=ignore_index, weight=loss_weights
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Compute the weighted cross-entropy loss
        return self.loss(logits, target)
    
    def __str__(self):
        return 'WeightedCrossEntropy'


class DICELoss(torch.nn.Module):
    def __init__(self, ignore_index: int) -> None:
        super(DICELoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        num_classes = logits.shape[1]

        # Convert logits to probabilities using softmax or sigmoid
        if num_classes == 1:
            probs = torch.sigmoid(logits)
        else:
            probs = F.softmax(logits, dim=1)

        # Create a mask to ignore the specified index
        mask = target != self.ignore_index
        target = target.clone()
        target[~mask] = 0

        # Convert target to one-hot encoding if necessary
        if num_classes == 1:
            target = target.unsqueeze(1)
        else:
            target = F.one_hot(target, num_classes=num_classes)
            target = target.permute(0, 3, 1, 2)

        # Apply the mask to the target
        target = target.float() * mask.unsqueeze(1).float()
        intersection = torch.sum(probs * target, dim=(2, 3))
        union = torch.sum(probs + target, dim=(2, 3))

        # Compute the Dice score
        dice_score = (2.0 * intersection + 1e-6) / (union + 1e-6)
        valid_dice = dice_score[mask.any(dim=1).any(dim=1)]
        dice_loss = 1 - valid_dice.mean()  # Dice loss is 1 minus the Dice score

        return dice_loss

    def __str__(self):
        return 'DICELoss'


class FocalLoss(torch.nn.Module):
    def __init__(self, ignore_index: int, distribution: list[float], gamma: float = 2.0) -> None:
        super(FocalLoss, self).__init__()
        # Initialize the weights based on the given distribution
        #self.weights = [1 / w if w!=0 else 0 for w in distribution]

        # Convert weights to a tensor and move to CUDA
        #loss_weights = torch.Tensor(self.weights).to("cuda")
        self.gamma = gamma
        self.loss = torch.nn.CrossEntropyLoss(
            ignore_index=ignore_index, reduction='none', # weight=loss_weights, 
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Compute the cross-entropy loss
        ce_loss = self.loss(logits, target)
        
        # Get the predicted probabilities
        probs = F.softmax(logits, dim=1)
        
        # Select the probabilities for the true class
        p_t = probs.gather(1, target.unsqueeze(1))  # Shape (B, 1)

        # Compute the focal loss component
        focal_loss = ce_loss * (1 - p_t) ** self.gamma
        
        # Return the mean loss over the batch
        return focal_loss.mean()
    
    def __str__(self):
        return 'FocalLoss'


class SupContrastiveLoss(torch.nn.Module):

    def __init__(self, tau=0.1):
        super().__init__()
        self.tau = tau
    
    def forward(self, projection, y):
        """This function generate the loss function based on SupContrast

        Args:
            projection (_type_): _description_
            y (_type_): _description_
        """
        correlation = (projection @ projection.T) / self.tau
        _max = torch.max(correlation, dim=1, keepdim=True)[0]

        exp_dot = torch.exp(correlation - _max) + 1e-7

        mask = (y.unsqueeze(1).repeat(1, len(y)) == y).to(projection.device)
        
        anchor_out = (1 - torch.eye(len(y))).to(projection.device)

        pij = mask * anchor_out # positives mask

        log_prob = -torch.log(
            exp_dot / torch.sum(exp_dot * anchor_out, dim=1, keepdim=True)
        )

        loss_samples = (
            torch.sum(log_prob * pij, dim=1) / (pij.sum(dim=1) + 1e-7)
        )

        return loss_samples.mean()

    def __str__(self):
        return 'SupContrastiveLoss'
    

class LogitCompensation(torch.nn.Module): 
    def __init__(self, distribution, ignore_index=-1, device="cuda"):
        super().__init__()
        priors = torch.tensor(distribution, dtype=torch.float32)
        self.log_priors = torch.log(priors).to(device)
        self.ignore_index = ignore_index

    def forward(self, seg_logits, seg_targets):
        """
        seg_logits: [N, C, H, W] raw logits from segmentation head
        seg_targets: [N, H, W] ground-truth labels
        """
        log_priors = self.log_priors

        # Add log-prior compensation to each class logit
        comp_logits = seg_logits + log_priors.view(1, -1, 1, 1)

        # Apply cross-entropy with ignore_index
        return F.cross_entropy(
            comp_logits, 
            seg_targets, 
            reduction='mean',
            ignore_index=self.ignore_index
        )

    def __str__(self):
        return 'LogitCompensation'


class CropConLoss(torch.nn.Module):
    def __init__(self, tau=0.1, ignore_index=-1, bcl_config="original", device='cuda'):
        super().__init__()
        self.temperature = tau
        self.ignore_index = ignore_index
        self.bcl_config = bcl_config
        self.device = device

    def forward(self, protos, proj2, target2, proj3, target3):
        if self.bcl_config == "original":
            return self.forward_original(protos, proj2, target2, proj3, target3)
        elif self.bcl_config == "class_balanced":
            return self.forward_class_balanced(protos, proj2, target2, proj3, target3)
        elif self.bcl_config == "prototypes":
            return self.forward_only_prototypes(protos, proj2, target2, proj3, target3)
        elif self.bcl_config == "decoupled":
            return self.forward_decoupled(protos, proj2, target2, proj3, target3)

    def forward_original(self, protos, proj2, target2, proj3, target3):
        # Normalize and combine features
        feats = F.normalize(torch.cat([proj2, proj3], dim=0), p=2, dim=-1)  # [M, D]
        labels = torch.cat([target2, target3], dim=0).long()                # [M]
        protos = F.normalize(protos, p=2, dim=-1)                           # [C, D]

        # Filter out ignored labels
        valid_mask = labels != self.ignore_index
        feats = feats[valid_mask]
        labels = labels[valid_mask]

        M, D = feats.shape
        if M == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        C = protos.size(0)

        # === Similarity matrices ===
        sim_matrix = torch.matmul(feats, feats.T) / self.temperature        # [M, M]
        proto_sim = torch.matmul(feats, protos.T) / self.temperature        # [M, C]

        # Remove self-similarity
        eye = torch.eye(M, device=self.device, dtype=torch.bool)
        sim_matrix = sim_matrix.masked_fill(eye, -float('inf'))

        # === Class match masks ===
        match_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)           # [M, M]

        # === Numerator ===
        numer_region = torch.exp(sim_matrix) * match_matrix                 # [M, M]
        numer_proto = torch.gather(torch.exp(proto_sim), 1, labels.view(-1,1))  # [M, 1]
        numer = numer_region.sum(dim=1) + numer_proto.squeeze(1)            # [M]

        # === Denominator with class balancing ===
        # Estimate class frequency from labels (both feats and protos)
        labels_all = torch.cat([labels, torch.arange(C, device=self.device)])    # [M + C]
        cls_freq = torch.bincount(labels_all, minlength=C).float()          # [C]
        cls_freq = cls_freq + 1e-6  # avoid division by zero

        # Construct per-instance weights
        feat_weights = cls_freq[labels]                                     # [M]
        proto_weights = cls_freq.unsqueeze(0).expand(M, -1)                 # [M, C]

        # Weight feat-feat similarities
        weight_matrix = feat_weights.unsqueeze(1).expand(-1, M)             # [M, M]
        weight_matrix = weight_matrix.masked_fill(eye, 1e6)                 # avoid self-similarities

        denom_region = torch.exp(sim_matrix) / weight_matrix                # [M, M]
        denom_proto = torch.exp(proto_sim) / proto_weights                  # [M, C]
        denom = denom_region.sum(dim=1) + denom_proto.sum(dim=1)            # [M]

        # === Final loss ===
        loss = -torch.log(numer / (denom + 1e-12))                          # [M]
        return loss.mean()
        
    def forward_class_balanced(self, protos, proj2, target2, proj3, target3):
        # Normalize and concatenate features
        feats = F.normalize(torch.cat([proj2, proj3], dim=0), p=2, dim=-1)  # [2B, D]
        labels = torch.cat([target2, target3], dim=0).long()                # [2B]
        protos = F.normalize(protos, p=2, dim=-1)                           # [C, D]

        B2 = feats.size(0)
        C = protos.size(0)

        # Extend features and labels with prototypes
        features_all = torch.cat([feats, protos], dim=0)                    # [2B + C, D]
        targets_all = torch.cat([
            labels,                    # [2B]
            torch.arange(C, device=self.device)  # [C]
        ], dim=0)                                                          # [2B + C]

        # Compute label frequencies in the batch (for weighting)
        batch_cls_count = torch.eye(C, device=self.device)[targets_all].sum(dim=0)  # [C]

        # Compute positive mask
        mask = torch.eq(labels.unsqueeze(1), targets_all.unsqueeze(0)).float()  # [2B, 2B+C]

        # Mask out self-similarities from region-region comparison
        logits_mask = torch.ones_like(mask)
        logits_mask[:, :B2].fill_diagonal_(0.0)  # [2B, 2B]

        # Compute cosine similarity logits
        logits = torch.matmul(feats, features_all.T) / self.temperature    # [2B, 2B+C]

        # Stability: subtract max
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # Compute weights for class balancing
        targets_expanded = targets_all.unsqueeze(0).expand(B2, -1)         # [2B, 2B+C]
        per_ins_weight = batch_cls_count[targets_expanded] - mask         # [2B, 2B+C]

        # Exponentiate logits and mask
        exp_logits = torch.exp(logits) * logits_mask

        # Compute class-balanced denominator
        denom = (exp_logits / (per_ins_weight + 1e-12)).sum(dim=1, keepdim=True)  # [2B, 1]

        # Compute log-probabilities
        log_prob = logits - torch.log(denom + 1e-12)                        # [2B, 2B+C]
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-12)  # [2B]

        # Final loss
        loss = -mean_log_prob_pos
        return loss.mean()
    
    def forward_only_prototypes(self, protos, proj2, target2, proj3, target3):
        # Normalize and combine features
        feats = F.normalize(torch.cat([proj2, proj3], dim=0), p=2, dim=-1)  # [M, D]
        labels = torch.cat([target2, target3], dim=0).long()                # [M]
        protos = F.normalize(protos, p=2, dim=-1)                           # [C, D]

        # Filter out ignored labels
        valid_mask = labels != self.ignore_index
        feats = feats[valid_mask]
        labels = labels[valid_mask]

        M, D = feats.shape
        if M == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        C = protos.size(0)

        labels_all = torch.cat([labels, torch.arange(C, device=self.device)]) 
        cls_freq = torch.bincount(labels_all, minlength=C).float()         
        cls_freq = cls_freq + 1e-12  # avoid division by zero
        proto_weights = cls_freq.unsqueeze(0).expand(M, -1)  

        # Similarity matrix
        proto_sim = torch.matmul(feats, protos.T) / self.temperature        # [M, C]

        # Numerator: class prototype
        numer_proto = torch.gather(torch.exp(proto_sim), 1, labels.view(-1,1))  # [M, 1]
        numer = numer_proto.squeeze(1)            # [M]

        # Denominator: all feats + all protos
        denom = (torch.exp(proto_sim) / proto_weights).sum(dim=1)              # [M]

        loss = -torch.log(numer / (denom + 1e-12))                          # [M]
        return loss.mean()

    def forward_decoupled(self, protos, proj2, target2, proj3, target3):
        feats = F.normalize(torch.cat([proj2, proj3], dim=0), p=2, dim=-1)  # [M, D]
        labels = torch.cat([target2, target3], dim=0).long()                # [M]
        protos = F.normalize(protos, p=2, dim=-1)                           # [C, D]

        valid_mask = labels != self.ignore_index
        feats = feats[valid_mask]
        labels = labels[valid_mask]

        M, D = feats.shape
        C = protos.size(0)
        
        sim_feats = torch.matmul(feats, feats.T) / self.temperature         # [M, M]
        sim_protos = torch.matmul(feats, protos.T) / self.temperature       # [M, C]

        eye = torch.eye(M, device=self.device, dtype=torch.bool)
        sim_feats = sim_feats.masked_fill(eye, -float('inf'))

        # Positive masks
        match_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)          # [M, M]
        proto_match = F.one_hot(labels, num_classes=C).bool()              # [M, C]

        # Numerators
        numer_feat = torch.exp(sim_feats) * match_matrix                   # [M, M]
        numer_proto = torch.exp(sim_protos)[proto_match].unsqueeze(1)     # [M, 1]
        numerator = numer_feat.sum(dim=1) + numer_proto.squeeze(1)
        numerator = torch.clamp(numerator, min=1e-12)

        # Denominator: exclude positives
        denom_feat = torch.exp(sim_feats) * (~match_matrix)
        denom_proto = torch.exp(sim_protos).masked_fill(proto_match, 0.0)
        denominator = denom_feat.sum(dim=1) + denom_proto.sum(dim=1)

        loss = -torch.log(numerator / (denominator + 1e-12))
        return loss.mean()

    def __str__(self):
        return 'CropConLoss'
