import torch
from torch.nn import functional as F
import torch.nn as nn
from typing import List, Tuple


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

def generate_same_image_mask(num_pixels_per_image: list[int], 
                             device: torch.device) -> torch.Tensor:
    """
    Genera una máscara que indica si dos píxeles pertenecen a la misma imagen.
    
    Args:
        num_pixels_per_image (List[int]): Lista con el número de píxeles
                                          en cada imagen del lote.
        device (torch.device): Dispositivo donde crear el tensor.

    Returns:
        torch.Tensor: Tensor de shape [1, num_total_pixels, num_total_pixels]
    """
    image_ids = []
    num_total_pixels = 0
    for img_id, pixel_count in enumerate(num_pixels_per_image):
        image_ids.extend([img_id] * pixel_count)
        num_total_pixels += pixel_count

    image_ids_tensor = torch.tensor(
        image_ids, dtype=torch.long, device=device
    ).view(num_total_pixels, 1)
    
    # Compara [N, 1] con [1, N] para obtener [N, N]
    same_image_mask = (image_ids_tensor == image_ids_tensor.t()).float()
    
    # Añade dimensión de lote: [1, N, N]
    return same_image_mask.unsqueeze(0)


def generate_ignore_mask(labels: torch.Tensor, 
                         ignore_labels: list[int]) -> torch.Tensor:
    """
    Genera máscara de ignorados (píxeles inválidos).
    
    Args:
        labels (torch.Tensor): Tensor de shape [B, N, 1] (píxeles aplanados).
        ignore_labels (List[int]): Lista de IDs de clase a ignorar.

    Returns:
        torch.Tensor: Tensor de shape [B, N, N]
    """
    # [B, N, 1]
    ignore_labels_tensor = torch.tensor(
        ignore_labels, dtype=labels.dtype, device=labels.device
    )
    
    # Compara [B, N, 1] con [len(ignore_labels)] -> [B, N, len(ignore_labels)]
    ignore_mask_per_pixel = (
        labels == ignore_labels_tensor.view(1, 1, -1)
    ).any(dim=2, keepdim=True) # [B, N, 1]

    # Un par (i, j) se ignora si *alguno* de los píxeles es inválido.
    # [B, N, 1] | [B, 1, N] -> [B, N, N]
    ignore_mask_matrix = (ignore_mask_per_pixel | 
                          ignore_mask_per_pixel.transpose(1, 2)).float()
    return ignore_mask_matrix


def generate_positive_and_negative_masks(
    labels: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Genera máscaras positiva (misma clase) y negativa (distinta clase).
    
    Args:
        labels (torch.Tensor): Tensor de shape [B, N, 1] (píxeles aplanados).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: positive_mask, negative_mask
                                           ambos de shape [B, N, N].
    """
    # Compara [B, N, 1] con [B, 1, N] -> [B, N, N]
    positive_mask = (labels == labels.transpose(1, 2)).float()
    negative_mask = 1.0 - positive_mask
    return positive_mask, negative_mask

# --- 3. Funciones de Aplanamiento y Cálculo de Loss ---

def collapse_spatial_dimensions(tensor_in: torch.Tensor) -> torch.Tensor:
    """
    Colapsa las dimensiones espaciales (H, W) en una sola (N).
    
    Args:
        tensor_in (torch.Tensor): Tensor de shape [B, C, H, W]
    
    Returns:
        torch.Tensor: Tensor de shape [B, N, C] donde N = H * W
    """
    b, c, h, w = tensor_in.shape
    # .view() -> [B, C, N]
    # .permute() -> [B, N, C]
    return tensor_in.view(b, c, -1).permute(0, 2, 1)


def compute_contrastive_loss(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    negative_mask: torch.Tensor,
    ignore_mask: torch.Tensor,
    epsilon: float = 1e-6
) -> torch.Tensor:
    """
    Cálculo de la pérdida contrastiva (InfoNCE modificada).
    
    Args:
        logits (torch.Tensor): Similitudes [B, N, N]
        positive_mask (torch.Tensor): Máscara de pares positivos [B, N, N]
        negative_mask (torch.Tensor): Máscara de pares negativos [B, N, N]
        ignore_mask (torch.Tensor): Máscara de pares a ignorar [B, N, N]
        epsilon (float): Pequeño valor para estabilidad numérica.

    Returns:
        torch.Tensor: Valor escalar de la pérdida.
    """
    validity_mask = 1.0 - ignore_mask
    positive_mask = positive_mask * validity_mask
    negative_mask = negative_mask * validity_mask

    # Numerador y denominador de la loss
    exp_logits = torch.exp(logits) * validity_mask

    # Denominador: exp(i) + sum(exp(j) para j en Negativos)
    denominator = exp_logits + torch.sum(
        exp_logits * negative_mask, dim=2, keepdim=True
    )
    
    # Probabilidad: exp(i) / (exp(i) + sum(negativos))
    # Usamos clamp() para evitar divisiones por cero (equiv a divide_no_nan)
    normalized_exp_logits = exp_logits / torch.clamp(denominator, min=epsilon)
    
    # -log(Probabilidad)
    # El truco (normalized_exp_logits * validity_mask + ignore_mask)
    # asegura que los píxeles ignorados tengan log(1) = 0
    neg_log_likelihood = -torch.log(
        normalized_exp_logits * validity_mask + ignore_mask + epsilon
    )

    # Normalizar por el número de positivos en la fila (dim 2)
    pos_sum_2 = torch.sum(positive_mask, dim=2, keepdim=True)
    normalized_weight_2 = positive_mask / torch.clamp(pos_sum_2, min=epsilon)
    
    neg_log_likelihood_sum_2 = torch.sum(
        neg_log_likelihood * normalized_weight_2, dim=2
    ) # [B, N]

    # Normalizar por el número de píxeles válidos (con al menos 1 positivo)
    # en el lote (dim 1)
    positive_mask_sum_1 = torch.sum(positive_mask, dim=2) # [B, N]
    valid_index = (positive_mask_sum_1 > 0).float() # [B, N]
    
    valid_index_sum_1 = torch.sum(valid_index, dim=1, keepdim=True) # [B, 1]
    normalized_weight_1 = valid_index / torch.clamp(valid_index_sum_1, min=epsilon)

    neg_log_likelihood_sum_1 = torch.sum(
        neg_log_likelihood_sum_2 * normalized_weight_1, dim=1
    ) # [B]

    loss = torch.mean(neg_log_likelihood_sum_1)
    return loss

# --- 4. Funciones de Pérdida Principales ---

def within_image_supervised_pixel_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    ignore_labels: list[int],
    temperature: float
) -> torch.Tensor:
    """
    Calcula la pérdida contrastiva SÓLO con píxeles de la misma imagen.
    
    Args:
        features (torch.Tensor): [B, N, C]
        labels (torch.Tensor): [B, N, 1]
        ignore_labels (List[int]): Clases a ignorar.
        temperature (float): Temperatura de la softmax.

    Returns:
        torch.Tensor: Pérdida escalar.
    """
    # Similitud entre todos los píxeles: [B, N, C] @ [B, C, N] -> [B, N, N]
    logits = torch.matmul(features, features.transpose(1, 2)) / temperature
    
    positive_mask, negative_mask = generate_positive_and_negative_masks(labels)
    ignore_mask = generate_ignore_mask(labels, ignore_labels)

    return compute_contrastive_loss(
        logits, positive_mask, negative_mask, ignore_mask
    )


def cross_image_supervised_pixel_contrastive_loss(
    features1: torch.Tensor,
    features2: torch.Tensor,
    labels1: torch.Tensor,
    labels2: torch.Tensor,
    ignore_labels: list[int],
    temperature: float
) -> torch.Tensor:
    """
    Calcula la pérdida contrastiva entre dos conjuntos de características/etiquetas
    (ej. original vs aumentada).
    
    Args:
        features1 (torch.Tensor): [B, N1, C]
        features2 (torch.Tensor): [B, N2, C]
        labels1 (torch.Tensor): [B, N1, 1]
        labels2 (torch.Tensor): [B, N2, 1]
        ignore_labels (List[int]): Clases a ignorar.
        temperature (float): Temperatura de la softmax.

    Returns:
        torch.Tensor: Pérdida escalar.
    """
    # N1 y N2 pueden ser diferentes si las imágenes originales y aumentadas
    # se redimensionan a tamaños distintos (aunque aquí N1=N2)
    batch_size, num_pixels1, _ = features1.shape
    _, num_pixels2, _ = features2.shape

    # Concatena a lo largo de la dimensión de píxeles
    # [B, N1+N2, C]
    features = torch.cat([features1, features2], dim=1)
    # [B, N1+N2, 1]
    labels = torch.cat([labels1, labels2], dim=1)

    num_pixels_list = [num_pixels1, num_pixels2]
    
    same_image_mask = generate_same_image_mask(
        num_pixels_list, device=features.device
    ) # [1, N1+N2, N1+N2]

    # Similitud [B, N1+N2, N1+N2]
    logits = torch.matmul(features, features.transpose(1, 2)) / temperature
    
    positive_mask, negative_mask = generate_positive_and_negative_masks(labels)
    # Filtra negativos: solo negativos de *diferentes* bloques
    negative_mask = negative_mask * same_image_mask
    
    ignore_mask = generate_ignore_mask(labels, ignore_labels)

    return compute_contrastive_loss(
        logits, positive_mask, negative_mask, ignore_mask
    )



class SupervisedPixelContrastiveLoss(torch.nn.Module):
    """
    Una implementación fiel en PyTorch de la Pixel-Wise Supervised Contrastive Loss
    del paper de Zhao et al. (2021) "Contrastive Learning for Label-Efficient
    Semantic Segmentation".

    Esta clase implementa tanto la variante "within-image" como la "cross-image".
    """

    def __init__(self, resize_size=128, temperature=0.1, ignore_index=-1, within_image=False):
        """
        Args:
            temperature (float): El parámetro de temperatura τ para escalar los logits.
            ignore_index (int): El valor en las etiquetas que debe ser ignorado durante el cálculo.
            loss_type (str): El tipo de pérdida a calcular. Opciones: 'within-image' o 'cross-image'.
        """
        super().__init__()
        self.temperature = temperature
        self.ignore_labels = [ignore_index]
        self.within_image_loss = within_image
        self.resize_size = resize_size//2

    def define_projector(self, proj_head):
        self.proj_head=proj_head

    def forward(
        self, 
        features_orig: torch.Tensor,
        features_aug: torch.Tensor,
        labels_orig: torch.Tensor,
        labels_aug: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calcula la pérdida contrastiva supervisada a nivel de píxel.
        
        Args:
            features_orig (torch.Tensor): [B, C_in, H_in, W_in]
            features_aug (torch.Tensor): [B, C_in, H_in, W_in]
            labels_orig (torch.Tensor): [B, 1, H_in, W_in] (tipo Long o Int)
            labels_aug (torch.Tensor): [B, 1, H_in, W_in] (tipo Long o Int)

        Returns:
            torch.Tensor: Pérdida escalar.
        """
        
        in_channels = features_orig.shape[1]
        device = features_orig.device
        
        # Redimensiona [B, C_in, H_in, W_in] -> [B, C_in, H_out, W_out]
        features_orig_resized = F.interpolate(
            features_orig, size=self.resize_size, mode='bilinear', align_corners=True
        )
        # Proyecta [B, C_in, H_out, W_out] -> [B, C_proj, H_out, W_out]
        features_orig_proj = self.proj_head(features_orig_resized)
        
        features_aug_resized = F.interpolate(
            features_aug, size=self.resize_size, mode='bilinear', align_corners=True
        )
        features_aug_proj = self.proj_head(features_aug_resized)

        # 3. Redimensionar Etiquetas
        # [B, 1, H_in, W_in] -> [B, 1, H_out, W_out]
        labels_orig_resized = F.interpolate(
            labels_orig.float(), size=self.resize_size, mode='nearest'
        ).long()
        
        labels_aug_resized = F.interpolate(
            labels_aug.float(), size=self.resize_size, mode='nearest'
        ).long()

        # 4. Colapsar dimensiones espaciales
        # [B, C_proj, H, W] -> [B, N, C_proj]
        features_orig_flat = collapse_spatial_dimensions(features_orig_proj)
        features_aug_flat = collapse_spatial_dimensions(features_aug_proj)
        
        # [B, 1, H, W] -> [B, N, 1]
        labels_orig_flat = collapse_spatial_dimensions(labels_orig_resized)
        labels_aug_flat = collapse_spatial_dimensions(labels_aug_resized)

        # 5. Calcular Pérdida
        
        if self.within_image_loss:
            loss_orig = within_image_supervised_pixel_contrastive_loss(
                features=features_orig_flat, 
                labels=labels_orig_flat,
                ignore_labels=self.ignore_labels, 
                temperature=self.temperature
            )
            loss_aug = within_image_supervised_pixel_contrastive_loss(
                features=features_aug_flat, 
                labels=labels_aug_flat,
                ignore_labels=self.ignore_labels, 
                temperature=self.temperature
            )
            return loss_orig + loss_aug

        # Lógica de Cross-Image
        batch_size = features_orig_flat.shape[0]
        
        # Barajar índices del lote (equiv. a tf.random.shuffle)
        shuffled_indices = torch.randperm(batch_size, device=device)
        
        # (equiv. a tf.gather)
        shuffled_features_aug = features_aug_flat[shuffled_indices]
        shuffled_labels_aug = labels_aug_flat[shuffled_indices]
        
        return cross_image_supervised_pixel_contrastive_loss(
            features1=features_orig_flat,
            features2=shuffled_features_aug,
            labels1=labels_orig_flat,
            labels2=shuffled_labels_aug,
            ignore_labels=self.ignore_labels,
            temperature=self.temperature
        )