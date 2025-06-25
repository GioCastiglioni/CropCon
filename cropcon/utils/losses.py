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


class BCLLoss(torch.nn.Module):
    def __init__(self, tau=0.1, ignore_index=-1, device='cuda'):
        super().__init__()
        self.temperature = tau
        self.ignore_index = ignore_index
        self.device = device

    def forward(self, protos, proj2, target2, proj3, target3):
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

        # Similarity matrices
        sim_matrix = torch.matmul(feats, feats.T) / self.temperature        # [M, M]
        proto_sim = torch.matmul(feats, protos.T) / self.temperature        # [M, C]

        # Remove self-similarity from sim_matrix
        eye = torch.eye(M, device=feats.device, dtype=torch.bool)
        sim_matrix = sim_matrix.masked_fill(eye, -float('inf'))

        # Class-equality mask
        match_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)           # [M, M]

        # Numerator: same-class samples + class prototype
        numer_region = torch.exp(sim_matrix) * match_matrix                 # [M, M]
        numer_proto = torch.gather(torch.exp(proto_sim), 1, labels.view(-1,1))  # [M, 1]
        numer = numer_region.sum(dim=1) + numer_proto.squeeze(1)            # [M]

        # Denominator: all feats + all protos
        denom = torch.exp(sim_matrix).sum(dim=1) + torch.exp(proto_sim).sum(dim=1)  # [M]

        loss = -torch.log(numer / (denom + 1e-12))                          # [M]
        return loss.mean()

    def forward_separated(self, protos, proj2, target2, proj3, target3):
        # Combine features and targets
        feats = F.normalize(torch.cat([proj2, proj3], dim=0), p=2, dim=-1)  # [M, D]
        labels = torch.cat([target2, target3], dim=0).long()                # [M]
        protos = F.normalize(protos, p=2, dim=-1)                           # [C, D]

        # Remove ignored labels
        valid_mask = labels != self.ignore_index
        feats = feats[valid_mask]
        labels = labels[valid_mask]

        M = feats.size(0)
        if M == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        C = protos.size(0)

        # === REGION-REGION TERM ===
        sim_matrix = torch.matmul(feats, feats.T) / self.temperature         # [M, M]
        eye = torch.eye(M, device=feats.device, dtype=torch.bool)
        sim_matrix.masked_fill_(eye, float('-inf'))                         # remove self-similarity

        log_probs_regions = F.log_softmax(sim_matrix, dim=1)                # [M, M]
        label_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)           # [M, M]
        pos_region_logprob = (log_probs_regions * label_matrix).sum(dim=1) # sum over matching samples

        # === REGION-PROTOTYPE TERM ===
        proto_sim = torch.matmul(feats, protos.T) / self.temperature         # [M, C]
        log_probs_protos = F.log_softmax(proto_sim, dim=1)                  # [M, C]
        pos_proto_logprob = log_probs_protos.gather(1, labels.view(-1,1)).squeeze(1)  # [M]

        # Final loss (sum of both)
        loss = -(pos_region_logprob + pos_proto_logprob)                    # [M]
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

        eye = torch.eye(M, device=feats.device, dtype=torch.bool)
        sim_feats = sim_feats.masked_fill(eye, -float('inf'))

        # Positive masks
        match_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)          # [M, M]
        proto_match = F.one_hot(labels, num_classes=C).bool()              # [M, C]

        # Numerators
        numer_feat = torch.exp(sim_feats) * match_matrix                   # [M, M]
        numer_proto = torch.exp(sim_protos)[proto_match].unsqueeze(1)     # [M, 1]
        numerator = numer_feat.sum(dim=1) + numer_proto.squeeze(1)

        # Denominator: exclude positives
        denom_feat = torch.exp(sim_feats) * (~match_matrix)
        denom_proto = torch.exp(sim_protos).masked_fill(proto_match, 0.0)
        denominator = denom_feat.sum(dim=1) + denom_proto.sum(dim=1)

        loss = -torch.log(numerator / (denominator + 1e-12))
        return loss.mean()

    def forward_separated_decoupled(self, protos, proj2, target2, proj3, target3):
        feats = F.normalize(torch.cat([proj2, proj3], dim=0), p=2, dim=-1)
        labels = torch.cat([target2, target3], dim=0).long()
        protos = F.normalize(protos, p=2, dim=-1)

        valid_mask = labels != self.ignore_index
        feats = feats[valid_mask]
        labels = labels[valid_mask]

        M = feats.size(0)
        if M == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        C = protos.size(0)

        # === REGION-REGION TERM ===
        sim_matrix = torch.matmul(feats, feats.T) / self.temperature
        eye = torch.eye(M, device=feats.device, dtype=torch.bool)
        sim_matrix = sim_matrix.masked_fill(eye, -float('inf'))

        label_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)
        neg_label_matrix = ~label_matrix & ~eye

        numer_region = torch.exp(sim_matrix) * label_matrix     # [M, M]
        denom_region = torch.exp(sim_matrix) * neg_label_matrix # [M, M]

        loss_region = -torch.log(
            numer_region.sum(dim=1) / (denom_region.sum(dim=1) + 1e-12)
        )

        # === REGION-PROTOTYPE TERM ===
        proto_sim = torch.matmul(feats, protos.T) / self.temperature
        proto_pos = torch.gather(torch.exp(proto_sim), 1, labels.view(-1,1)).squeeze(1)  # [M]

        proto_mask = F.one_hot(labels, C).bool()  # [M, C]
        proto_neg = torch.exp(proto_sim).masked_fill(proto_mask, 0.0).sum(dim=1)

        loss_proto = -torch.log(proto_pos / (proto_neg + 1e-12))  # [M]

        return (loss_region + loss_proto).mean()

    def __str__(self):
        return 'BCLLoss'
