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

class VICReg(torch.nn.Module):

    def __init__(self, vic_weights: list[float], inv_loss: str = "mse", ignore_index = None):
        super().__init__()

        self.variance_loss_epsilon = 1e-08
        
        self.variance_loss_weight = vic_weights[0]
        self.invariance_loss_weight = vic_weights[1]
        self.covariance_loss_weight = vic_weights[2]

        if inv_loss == "mse":
            self.inv = torch.nn.MSELoss()
        elif inv_loss == "cca":
            self.inv = CCALoss()
        elif inv_loss == "ntxent":
            self.inv = NTXentLoss()

    def forward(self, z_a, z_b, each_comp=False):

        loss_inv = self.inv(z_a, z_b)

        std_z_a = torch.sqrt(
            z_a.var(dim=0, unbiased=False) + self.variance_loss_epsilon
        )
        std_z_b = torch.sqrt(
            z_b.var(dim=0, unbiased=False) + self.variance_loss_epsilon
        )
        loss_v_a = torch.mean(F.relu(1 - std_z_a))
        loss_v_b = torch.mean(F.relu(1 - std_z_b))
        loss_var = loss_v_a + loss_v_b

        N, D = z_a.shape

        z_a = z_a - z_a.mean(dim=0)
        z_b = z_b - z_b.mean(dim=0)

        cov_z_a = ((z_a.T @ z_a) / N).square()  # DxD
        cov_z_b = ((z_b.T @ z_b) / N).square()  # DxD
        loss_c_a = (cov_z_a.sum() - cov_z_a.diagonal().sum()) / D
        loss_c_b = (cov_z_b.sum() - cov_z_b.diagonal().sum()) / D
        loss_cov = loss_c_a + loss_c_b

        
        weighted_var = loss_var * self.variance_loss_weight
        weighted_cov = loss_cov * self.covariance_loss_weight

        weighted_inv = loss_inv * self.invariance_loss_weight

        loss = weighted_inv + weighted_var + weighted_cov
        if each_comp: return loss.mean(), loss_var, loss_inv, loss_cov
        else: return loss.mean()


class SimCLR(torch.nn.Module):
    def __init__(self, tau: float = 0.1):
        super().__init__()
        self.temperature = tau

    def forward(self, z_a, z_b):
        """
        z_a: [N, D] tensor
        z_b: [N, D] tensor
        """

        N = z_a.shape[0]
        # Normalize representations
        z_a = F.normalize(z_a, dim=1)
        z_b = F.normalize(z_b, dim=1)

        # Concatenate for 2N samples
        z = torch.cat([z_a, z_b], dim=0)  # [2N, D]

        # Compute similarity matrix
        sim = torch.matmul(z, z.T) / self.temperature  # [2N, 2N]

        # Mask self-similarity
        mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, -float("inf"))

        # Positive pairs: i with i+N (first with second view)
        targets = torch.arange(N, device=z.device)
        targets = torch.cat([targets + N, targets], dim=0)  # [2N]

        # Cross-entropy loss
        loss = F.cross_entropy(sim, targets)

        return loss


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
        self.ignore_index = ignore_index

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


class SupConLoss(torch.nn.Module):

    def __init__(self, tau=0.1, ignore_index=None):
        super().__init__()
        self.tau = tau
        self.ignore_index = ignore_index
    
    def forward(self, projection, y):
        """
        Supervised Contrastive Loss (Khosla et al.) with ignore_index.

        Args:
            projection (Tensor): shape (N, D), normalized or unnormalized embeddings
            y (Tensor): shape (N,), labels
        """
        device = projection.device
        n = len(y)

        # mask out ignored samples
        if self.ignore_index is not None:
            valid_mask = (y != self.ignore_index)
            projection = projection[valid_mask]
            y = y[valid_mask]
            n = len(y)

        if n <= 1:  # nothing to contrast
            return torch.tensor(0.0, device=device, requires_grad=True)

        # similarity
        correlation = (projection @ projection.T) / self.tau
        _max = torch.max(correlation, dim=1, keepdim=True)[0]
        exp_dot = torch.exp(correlation - _max) + 1e-7

        # positive mask (same class, excluding self)
        mask = (y.unsqueeze(1) == y.unsqueeze(0)).to(device)
        anchor_out = (1 - torch.eye(n, device=device))
        pij = mask * anchor_out  # positives mask

        # log-probs
        log_prob = -torch.log(
            exp_dot / (torch.sum(exp_dot * anchor_out, dim=1, keepdim=True) + 1e-7)
        )

        # per-sample loss (average over positives for each anchor)
        loss_samples = torch.sum(log_prob * pij, dim=1) / (pij.sum(dim=1) + 1e-7)

        return loss_samples.mean()

    def __str__(self):
        return 'SupConLoss'
    

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

    def forward(self, protos, proj2, target2):

        feats = F.normalize(proj2, p=2, dim=-1)
        labels = target2.long()
        protos = F.normalize(protos, p=2, dim=-1)
        
        return self.forward_original(protos, feats, labels)

    def forward_original(self, protos, feats, labels):                       # [C, D]

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

        # === Prototypes Regularization ===
        prot_var_reg = torch.sqrt(protos.var(dim=0) + 1e-12)
        prot_var_reg = torch.mean(F.relu(1 - prot_var_reg))

        prot_cov_reg = ((protos.T @ protos) / (C - 1)).square()
        prot_cov_reg = (prot_cov_reg.sum() - prot_cov_reg.diagonal().sum()) / D

        return loss.mean() + prot_var_reg + 0.1 * prot_cov_reg

    def __str__(self):
        return 'CropConLoss'
