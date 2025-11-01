import torch
from torch.nn import functional as F
import torch.nn as nn
from typing import List, Tuple
import torch.distributed as dist


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



class JepaLoss(nn.Module):
    def __init__(self, d_model, nhead=8, num_decoder_layers=6, grid_size=8):
        super().__init__()
        self.d_model = d_model
        self.grid_size = (grid_size, grid_size)
        self.num_patches = grid_size ** 2 

        self.pos_encoding = nn.Parameter(torch.zeros(1, d_model, self.grid_size[0], self.grid_size[1]))
        nn.init.trunc_normal_(self.pos_encoding, std=.02) 

        self.predictor_queries = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.predictor_queries, std=.02)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.predictor = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )

        self.loss_fn = nn.MSELoss()

    def forward(self, student_features, teacher_features, idx_context, idx_target):

        B, D, H, W = student_features.shape
        N_tgt = idx_target.shape[1]
        
        if H != self.grid_size[0] or W != self.grid_size[1]:
            raise ValueError(f"Feature map grid ({H}, {W}) does not match grid_size {self.grid_size}")

        student_features_pos = student_features + self.pos_encoding
        teacher_features_pos = teacher_features + self.pos_encoding

        student_seq = student_features_pos.flatten(2).permute(0, 2, 1)
        teacher_seq = teacher_features_pos.flatten(2).permute(0, 2, 1)
        
        pos_encoding_flat = self.pos_encoding.flatten(2).permute(0, 2, 1)

        idx_target_expanded = idx_target.unsqueeze(-1).expand(-1, -1, D)
        
        target_tokens = teacher_seq.gather(dim=1, index=idx_target_expanded).detach()

        idx_context_expanded = idx_context.unsqueeze(-1).expand(-1, -1, D)
        context_tokens = student_seq.gather(dim=1, index=idx_context_expanded)
        query_tokens = self.predictor_queries.expand(B, N_tgt, -1)
        
        query_pos = pos_encoding_flat.expand(B, -1, -1).gather(dim=1, index=idx_target_expanded)
        
        queries_with_pos = query_tokens + query_pos

        predictions = self.predictor(
            tgt=queries_with_pos,
            memory=context_tokens
        )
        loss = self.loss_fn(predictions, target_tokens)
        
        return loss


@torch.no_grad()
def phi_gain(a, b):
    """
    Calcula la función de ganancia phi(a, b) = b - a + a * log(a / b)
    """
    return b - a + a * torch.log(a / (b + 1e-8) + 1e-8)

@torch.no_grad()
def greedy_sinkhorn(P, X, kappa, iterations):
    """
    Argumentos:
    - P (torch.Tensor): Prototipos de la clase c. Forma: [K, D]
    - X (torch.Tensor): Píxeles de la clase c. Forma: [N, D]
    - kappa (float): Parámetro de suavizado (temperatura).
    - iterations (int): Número de iteraciones.
    
    Devuelve:
    - L (torch.Tensor): Matriz de asignación de transporte óptimo. Forma: [K, N]
    """
    
    K, D = P.shape
    N = X.shape[0]
    
    # 1. Calcular la matriz de similitud (costo)
    sim = P @ X.T
    A = torch.exp(sim / kappa)  # Matriz A [K, N]

    # 2. Definir las marginales objetivo (restricciones de Eq. 9)
    r_target = torch.full((K,), N / K, device=P.device, dtype=P.dtype)
    c_target = torch.full((N,), 1.0, device=P.device, dtype=P.dtype)

    # 3. Inicializar los vectores de escala u y v
    u = torch.ones(K, device=P.device, dtype=P.dtype)
    v = torch.ones(N, device=P.device, dtype=P.dtype)

    # 4. Bucle de iteraciones de Greedy Sinkhorn
    for _ in range(iterations):
        r_curr = u * (A @ v)
        c_curr = v * (A.T @ u)

        phi_r = phi_gain(r_curr, r_target)
        phi_c = phi_gain(c_curr, c_target)

        max_phi_r, idx_r = torch.max(phi_r, dim=0)
        max_phi_c, idx_c = torch.max(phi_c, dim=0)

        # 5. Actualizar solo el vector de escala más crítico
        if max_phi_r > max_phi_c:
            u[idx_r] = r_target[idx_r] / (A[idx_r, :] @ v + 1e-8)
        else:
            v[idx_c] = c_target[idx_c] / (A[:, idx_c] @ u + 1e-8)

    # 6. Calcular la matriz de transporte final L
    L = u.unsqueeze(1) * A * v.unsqueeze(0)
    
    return L

class PrototypeBasedSemSegLoss(nn.Module):
    
    def __init__(self, 
                 num_classes, 
                 num_prototypes_per_class, 
                 feature_dim, 
                 ignore_index=255,
                 momentum=0.999,      # mu para Eq. 14 [cite: 328]
                 sk_kappa=0.05,       # kappa para Eq. 9 [cite: 273]
                 sk_iterations=50,
                 ppc_temperature=0.1, # tau para Eq. 11 
                 lambda_ce=1.0,
                 lambda_ppc=0.01,     # lambda_1 en Eq. 13 [cite: 305]
                 lambda_ppd=0.01):    # lambda_2 en Eq. 13 [cite: 305]
        
        super().__init__()
        
        self.C = num_classes
        self.K = num_prototypes_per_class
        self.D = feature_dim
        self.ignore_index = ignore_index
        
        self.momentum = momentum
        self.sk_kappa = sk_kappa
        self.sk_iterations = sk_iterations
        self.ppc_temperature = ppc_temperature
        
        self.lambda_ce = lambda_ce
        self.lambda_ppc = lambda_ppc
        self.lambda_ppd = lambda_ppd
        
        # Inicializar los prototipos
        # Los registramos como un 'buffer', no como 'parameter'
        # para que no sean actualizados por el optimizador (SGD)
        prototypes = torch.randn(self.C, self.K, self.D)
        prototypes = F.normalize(prototypes, p=2, dim=2)
        self.register_buffer("prototypes", prototypes)

    def _get_dist_info(self):
        """Helper para DDP"""
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1
        return rank, world_size
    
    @torch.no_grad()
    def _compute_cluster_averages(self, P_c, X_c):
        """
        Calcula los promedios de clúster \bar{i}_c,k para una clase c.
        Devuelve:
        - avg_vectors (Tensor[K, D]): Vectores promedio. Ceros si clúster vacío.
        - counts (Tensor[K]): Conteo de píxeles por clúster.
        - assignments (Tensor[N_c]): Índice 'k' (0 a K-1) asignado a cada píxel.
        """
        N_c = X_c.shape[0]
        if N_c == 0:
            avg_vectors = torch.zeros_like(P_c)
            counts = torch.zeros(self.K, device=P_c.device, dtype=torch.float)
            assignments = torch.zeros(0, device=P_c.device, dtype=torch.long)
            return avg_vectors, counts, assignments

        # 1. Resolver clustering
        L = greedy_sinkhorn(P_c, X_c, self.sk_kappa, self.sk_iterations) # [K, N_c]

        # 2. Obtener asignaciones
        assignments = torch.argmax(L, dim=0)  # [N_c]

        # 3. Calcular promedios
        sum_vectors = torch.zeros_like(P_c)
        counts = torch.bincount(assignments, minlength=self.K).float() # [K]
        
        idx_expanded = assignments.unsqueeze(1).expand(-1, self.D)
        sum_vectors.scatter_add_(0, idx_expanded, X_c)

        # Calcular promedio y l2-normalizar
        avg_vectors = sum_vectors / (counts.unsqueeze(1) + 1e-8)
        avg_vectors = F.normalize(avg_vectors, p=2, dim=1)
        
        # Asegurar que clústeres vacíos (0/EPSILON) sean ceros
        avg_vectors[counts == 0] = 0.0
        
        return avg_vectors, counts, assignments

    
    def forward(self, features, labels):
        """
        Entrada:
        - features (Tensor[B, D, H, W]): Embeddings de píxeles (l2-norm NO requerida)
        - labels (Tensor[B, H, W]): Etiquetas de clase (ground truth)
        """
        
        rank, world_size = self._get_dist_info()
        
        # --- 0. Preparación ---
        B, D, H, W = features.shape
        assert D == self.D, "La dimensión de features no coincide"

        # Normalizar features
        features_norm = F.normalize(features, p=2, dim=1)
        
        # Aplanar features y etiquetas
        features_flat = features_norm.permute(0, 2, 3, 1).contiguous().view(-1, D) # [N_total, D]
        labels_flat = labels.view(-1) # [N_total]
        
        # Crear máscara para píxeles válidos (ignorar 'ignore_label')
        valid_mask = (labels_flat != self.ignore_index)
        labels_valid = labels_flat[valid_mask]   # [N_valid]
        features_valid = features_flat[valid_mask] # [N_valid]
        
        N_valid = features_valid.shape[0]
        if N_valid == 0:
            # Si no hay píxeles válidos en este batch (raro), devolver 0
            return features.sum() * 0.0

        # --- 1. Cálculo de L_CE (Eq. 7) ---
        # "Logits" son la similitud con el prototipo MÁS CERCANO de cada clase
        
        # Calcular todas las similitudes [N_valid, C*K]
        all_prototypes_flat = self.prototypes.view(self.C * self.K, self.D)
        sim_all = features_valid @ all_prototypes_flat.T
        
        # Encontrar la similitud MÁXIMA por clase [N_valid, C, K] -> [N_valid, C]
        sim_per_class, _ = sim_all.view(N_valid, self.C, self.K).max(dim=2)
        
        # Los logits para CE son las similitudes máximas
        # Eq. 6/7 usa `s_i,c` como *distancia*, `p(c|i) = exp(-s_i,c)`
        # Usar `s_i,c = -sim_per_class`
        # `logits = -s_i,c = sim_per_class`
        # (Nota: El paper no usa temperatura aquí, pero a veces se añade.
        # Seguiremos el paper.)
        logits_ce = sim_per_class # [N_valid, C]
        
        loss_ce = F.cross_entropy(logits_ce, labels_valid)

        # --- 2. Clustering y Asignación ---
        
        # Tensores para almacenar los resultados del clustering
        local_i_bar_k_all = torch.zeros_like(self.prototypes)
        local_counts_all = torch.zeros(self.C, self.K, device=features.device, dtype=torch.float)
        
        # Almacena el índice 'k' (0 a K-1) asignado a cada píxel
        pixel_assigned_k_idx = torch.zeros_like(labels_valid)

        present_classes = torch.unique(labels_valid)
        
        for c in present_classes:
            c = c.item()
            class_mask = (labels_valid == c)
            features_c = features_valid[class_mask]
            prototypes_c = self.prototypes[c]
            
            # Realizar clustering para la clase c
            i_bar_k_c, counts_c, assignments_c = self._compute_cluster_averages(
                prototypes_c, features_c
            )
            
            local_i_bar_k_all[c] = i_bar_k_c
            local_counts_all[c] = counts_c
            pixel_assigned_k_idx[class_mask] = assignments_c
            
        # --- 3. Cálculo de L_PPC y L_PPD (Usando asignaciones locales) ---
        
        # Obtener el prototipo "positivo" para cada píxel [N_valid, D]
        # (El asignado por Sinkhorn)
        positive_prototypes = self.prototypes[labels_valid, pixel_assigned_k_idx]
        
        # --- L_PPD (Eq. 12) ---
        # Similitud Coseno con el prototipo positivo
        sim_positive = (features_valid * positive_prototypes).sum(dim=1)
        loss_ppd = (1 - sim_positive).pow(2).mean()

        # --- L_PPC (Eq. 11) ---
        # Esto es un Cross-Entropy contra TODOS los prototipos
        
        # 'logits' son las similitudes con TODOS los prototipos
        # Ya los calculamos en el paso 1: sim_all [N_valid, C*K]
        logits_ppc = sim_all / self.ppc_temperature
        
        # 'target' es el índice plano (0 a C*K - 1) del prototipo positivo
        target_ppc = labels_valid * self.K + pixel_assigned_k_idx
        
        loss_ppc = F.cross_entropy(logits_ppc, target_ppc) 

        # --- 4. Pérdida Total (Eq. 13) ---
        total_loss = (
            self.lambda_ce * loss_ce + 
            self.lambda_ppc * loss_ppc + 
            self.lambda_ppd * loss_ppd
        )

        # --- 5. Sincronización DDP y Actualización de Prototipos (Eq. 14) ---
        
        # Para la actualización, necesitamos los promedios GLOBALES
        # Calculamos la suma (promedio * conteo) y los conteos
        local_sum_vectors = local_i_bar_k_all * local_counts_all.unsqueeze(-1)
        local_counts = local_counts_all
        
        if world_size > 1:
            dist.all_reduce(local_sum_vectors, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_counts, op=dist.ReduceOp.SUM)
            
        # Calcular promedios globales
        global_avg_vectors = local_sum_vectors / (local_counts.unsqueeze(-1) + 1e-8)
        global_avg_vectors = F.normalize(global_avg_vectors, p=2, dim=-1)
        global_avg_vectors[local_counts == 0] = 0.0
        
        # Actualizar prototipos con Ecuación 14
        with torch.no_grad():
            valid_mask = local_counts > 0 # Solo actualizar prototipos que recibieron píxeles

            updated_values = (
                self.momentum * self.prototypes[valid_mask] + 
                (1 - self.momentum) * global_avg_vectors[valid_mask]
            )
            # Re-normalizar por si acaso
            updated_values = F.normalize(updated_values, p=2, dim=-1)
            
            self.prototypes.data[valid_mask] = updated_values

        return total_loss 

    def __str__(self):
        return 'PrototypeBasedSemSegLoss'




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

