import torch
from torch.nn import functional as F
import torch.nn as nn
import torch.distributed as dist
from torch.distributed._functional_collectives import (
    all_reduce as functional_all_reduce,
)
from torch.distributed.nn import ReduceOp


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
            dim_feedforward=d_model * 2,
            dropout=1/3,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.predictor = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )

        self.v = 2
        self.i = 5
        self.c = 1

    def forward(self, student_features, teacher_features, mask_ctx_1d, mask_tgt_1d):
        
        B, D, H, W = student_features.shape
        if H != self.grid_size[0] or W != self.grid_size[1]:
            raise ValueError(f"Feature map grid ({H}, {W}) does not match grid_size {self.grid_size}")
        
        var_loss = torch.tensor(0.0, device=student_features.device)
        cov_loss = torch.tensor(0.0, device=student_features.device)

        if B >= 2:
            
            z_avg = student_features.mean(dim=[2, 3]) # Shape: [B, D]
            
            std = torch.sqrt(z_avg.var(dim=0) + 1e-5) # Shape: [D]
            var_loss = torch.mean(F.relu(1 - std))

            z_all_vectors = student_features.flatten(2).permute(0, 2, 1).flatten(0, 1) # [B*H*W, D]
            N_vectors = z_all_vectors.shape[0] # N_vectors = B * H * W

            z_centered = z_all_vectors - z_all_vectors.mean(dim=0) # Shape: [N_vectors, D]
            
            cov = (z_centered.T @ z_centered) / (N_vectors - 1) # Shape: [D, D]
            
            cov_loss = cov.fill_diagonal_(0).pow(2).sum() / D

        student_features_pos = student_features + self.pos_encoding
        teacher_features_pos = teacher_features + self.pos_encoding
        
        student_seq_all_batch = student_features_pos.flatten(2).permute(0, 2, 1)
        teacher_seq_all_batch = teacher_features_pos.flatten(2).permute(0, 2, 1)
        
        pos_encoding_flat = self.pos_encoding.flatten(2).permute(0, 2, 1).squeeze(0)

        total_loss = 0.0
        
        for b in range(B):
            student_seq_all = student_seq_all_batch[b]
            teacher_seq_all = teacher_seq_all_batch[b]
            ctx_mask_b = mask_ctx_1d[b]
            tgt_mask_b = mask_tgt_1d[b]

            context_tokens = student_seq_all[ctx_mask_b] 
            target_tokens = teacher_seq_all[tgt_mask_b].detach() 
            
            if target_tokens.shape[0] == 0 or context_tokens.shape[0] == 0:
                continue
                
            context_pos = pos_encoding_flat[ctx_mask_b]
            target_pos = pos_encoding_flat[tgt_mask_b]
            
            query_tokens = self.predictor_queries.expand(target_tokens.shape[0], 1, -1).squeeze(1)
            
            memory_with_pos = context_tokens + context_pos
            queries_with_pos = query_tokens + target_pos

            predictions = self.predictor(
                tgt=queries_with_pos.unsqueeze(0),
                memory=memory_with_pos.unsqueeze(0)
            ).squeeze(0) 
            
            loss_b = F.mse_loss(predictions, target_tokens)
            total_loss += loss_b
        
        if B > 0:
            mse_loss = total_loss / B
        else:
            mse_loss = torch.tensor(0.0, device=student_features.device)

        final_loss = self.i * mse_loss + self.v * var_loss + self.c * cov_loss

        return {
            "loss": final_loss,
            "mse_loss": mse_loss.detach(),
            "var_loss": var_loss.detach(),
            "cov_loss": cov_loss.detach()
        }
        
    def __str__(self):
        return "JEPALoss"


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
    

class LogitCompensation(nn.Module):
    def __init__(self, distribution):
        super(LogitCompensation, self).__init__()
        
        priors = torch.tensor(distribution).float()
        logit_adj = torch.log(priors + 1e-9) 
        self.register_buffer('logit_adj', logit_adj)

    def forward(self, logits, targets):
        
        #logits = logits + self.logit_adj.view(1, -1, 1, 1)
        loss = F.cross_entropy(logits, targets)
        
        return loss
    
    def __str__(self):
        return 'LogitCompensation'


class BCLSegmentationLoss(nn.Module):
    def __init__(self, num_classes, tau=0.1, max_anchors=1024, max_context=2048, ignore_index=-1):
        """
        Args:
            num_classes (int): Cantidad de clases.
            tau (float): Temperatura.
            max_anchors (int): Cuantos píxeles muestrear de v1 para calcular la loss (reduce memoria).
            max_context (int): Cuantos píxeles de (v1 + v2) usar como "diccionario" o contexto.
            ignore_index (int): Label a ignorar.
        """
        super(BCLSegmentationLoss, self).__init__()
        self.num_classes = num_classes
        self.tau = tau
        self.max_anchors = max_anchors
        self.max_context = max_context
        self.ignore_index = ignore_index

    def _sample_pixels(self, features, targets, num_samples):
        b, d, h, w = features.shape
        feat_flat = features.permute(0, 2, 3, 1).reshape(-1, d)
        targ_flat = targets.reshape(-1)
        
        mask = targ_flat != self.ignore_index
        feat_valid = feat_flat[mask]
        targ_valid = targ_flat[mask]
        
        if feat_valid.size(0) == 0:
            return None, None

        if feat_valid.size(0) > num_samples:
            perm = torch.randperm(feat_valid.size(0), device=features.device)[:num_samples]
            return feat_valid[perm], targ_valid[perm]
        else:
            return feat_valid, targ_valid

    def forward(self, z1, z2, prototypes, targets1, targets2):
        device = z1.device
        
        anchors, anchors_labels = self._sample_pixels(z1, targets1, self.max_anchors)
        
        if anchors is None:
            return 0.0*z1.sum()

        ctx1, ctx1_labels = self._sample_pixels(z1, targets1, self.max_context // 2)
        ctx2, ctx2_labels = self._sample_pixels(z2, targets2, self.max_context // 2)
        
        # Manejo de casos vacíos en contexto
        ctx_list = []
        lbl_list = []
        if ctx1 is not None: 
            ctx_list.append(ctx1); lbl_list.append(ctx1_labels)
        if ctx2 is not None: 
            ctx_list.append(ctx2); lbl_list.append(ctx2_labels)
            
        if not ctx_list:
            return torch.tensor(0.0, device=device, requires_grad=True)

        context_features = torch.cat(ctx_list, dim=0)
        context_labels = torch.cat(lbl_list, dim=0)

        anchors = F.normalize(anchors, dim=1)
        context_features = F.normalize(context_features, dim=1)
        prototypes = F.normalize(prototypes, dim=1)

        pool_features = torch.cat([context_features, prototypes], dim=0)
        
        proto_labels = torch.arange(self.num_classes, device=device)
        pool_labels = torch.cat([context_labels, proto_labels], dim=0)

        sim_matrix = torch.matmul(anchors, pool_features.T) / self.tau
        exp_sim = torch.exp(sim_matrix)

        pool_one_hot = F.one_hot(pool_labels, num_classes=self.num_classes).float()
        sum_exp_per_class = torch.matmul(exp_sim, pool_one_hot)
        cardinality = pool_one_hot.sum(dim=0).clamp(min=1.0)
        avg_exp_per_class = sum_exp_per_class / cardinality.view(1, -1)

        bcl_denominator = avg_exp_per_class.sum(dim=1, keepdim=True)
        
        log_prob_matrix = sim_matrix - torch.log(bcl_denominator + 1e-9)
        mask_positives = (anchors_labels.unsqueeze(1) == pool_labels.unsqueeze(0)).float()
        log_probs_pos = (log_prob_matrix * mask_positives).sum(dim=1)
        
        num_positives = mask_positives.sum(dim=1).clamp(min=1.0)
        loss_per_anchor = - (log_probs_pos / num_positives)
        
        return loss_per_anchor.mean()
    

class BalancedContrastiveLearning(nn.Module):
    def __init__(
            self,
            num_classes,
            distribution,
            ignore_index=-1,
            lamb=2.0,
            mu=0.6,
            temperature=0.1,
            in_channels=64,
            hidden_d=512,
            out_d=128
        ):
        super(BalancedContrastiveLearning, self).__init__()
        self.num_classes = num_classes
        self.distribution = distribution
        self.ignore_index = ignore_index
        self.lamb = lamb
        self.mu = mu
        self.temperature = temperature
        self.in_channels = in_channels
        self.hidden_d = hidden_d
        self.out_d = out_d

        self.LC = LogitCompensation(self.distribution)
        self.BCL = BCLSegmentationLoss(
            self.num_classes, 
            tau=self.temperature,
            max_anchors=4096, 
            max_context=32768,
            ignore_index=self.ignore_index
        )

        #self.views_mlp(in_channels, hidden_d, out_d)
        #self.prot_mlp(in_channels, hidden_d, out_d)

    def forward(self, logits, z2, z3, targets, targets2, targets3, prototypes):

        LC = self.LC(logits, targets)

        z2 = self.views_mlp(z2)
        z3 = self.views_mlp(z3)
        prototypes = self.prot_mlp(prototypes).flatten(start_dim=1)

        BCL = self.BCL(z2, z3, prototypes, targets2, targets3)
        
        return self.lamb*LC + self.mu*BCL
    
    def __str__(self):
        return 'BalancedContrastiveLearning'
    

class LeJEPA(nn.Module):
    def __init__(
            self,
            num_slices,
            lamb=0.05,
            knots=17,
        ):
        super(LeJEPA, self).__init__()
        self.num_slices = num_slices
        self.lamb = lamb
        self.knots = knots
        self.sigreg = SlicingUnivariateTest(EppsPulley(n_points=self.knots), num_slices=self.num_slices)

    def forward(self, global_views, local_views):
        B, K = global_views[0].shape
        global_centroid = torch.stack(global_views).mean(dim=0)
        all_views_list = global_views + local_views
        all_views_tensor = torch.stack(all_views_list)
        inv = (global_centroid.unsqueeze(0) - all_views_tensor).square().mean()
        sigreg = torch.stack([self.sigreg(view) for view in all_views_list]).mean()
        return (1-self.lamb)*inv + self.lamb*sigreg
    
    def __str__(self):
        return 'LeJEPA'


def all_reduce(x, op="AVG"):
    if dist.is_available() and dist.is_initialized():
        return functional_all_reduce(x, op.lower(), dist.group.WORLD)
    else:
        return x


class SlicingUnivariateTest(torch.nn.Module):

    def __init__(
        self,
        univariate_test,
        num_slices: int,
        reduction: str = "mean",
        sampler: str = "gaussian",
        clip_value: float = None,
    ):
        super().__init__()
        self.reduction = reduction
        self.num_slices = num_slices
        self.sampler = sampler
        self.univariate_test = univariate_test
        self.clip_value = clip_value
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))

        # Generator reuse
        self._generator = None
        self._generator_device = None

    def _get_generator(self, device, seed):
        """Get or create generator for given device and seed."""
        if self._generator is None or self._generator_device != device:
            self._generator = torch.Generator(device=device)
            self._generator_device = device
        self._generator.manual_seed(seed)
        return self._generator

    def forward(self, x):
        """
        Apply sliced univariate test to multivariate samples.
        Args:
            x (torch.Tensor): Input samples of shape (*, N, D) where * represents
                any number of batch dimensions, N is the number of samples, and
                D is the feature dimension.
        Returns:
            torch.Tensor: Aggregated test statistic(s).
                - Scalar tensor if reduction='mean' or 'sum'
                - Shape (*, num_slices) if reduction=None
        """
        with torch.no_grad():
            # Synchronize global_step across all ranks
            global_step_sync = all_reduce(self.global_step.clone(), op="MAX")
            seed = global_step_sync.item()
            dev = dict(device=x.device)

            # Get reusable generator
            g = self._get_generator(x.device, seed)

            proj_shape = (x.size(-1), self.num_slices)
            A = torch.randn(proj_shape, **dev, generator=g)
            A /= A.norm(p=2, dim=0)
            self.global_step.add_(1)

        stats = self.univariate_test(x @ A)
        if self.clip_value is not None:
            stats[stats < self.clip_value] = 0
        if self.reduction == "mean":
            return stats.mean()
        elif self.reduction == "sum":
            return stats.sum()
        elif self.reduction is None:
            return stats
        
class UnivariateTest(torch.nn.Module):
    def __init__(self, eps: float = 1e-5, sorted: bool = False):
        super().__init__()
        self.eps = eps
        self.sorted = sorted
        self.g = torch.distributions.normal.Normal(0, 1)

    def prepare_data(self, x):
        if self.sorted:
            s = x
        else:
            s = x.sort(descending=False, dim=-2)[0]
        return s

    def dist_mean(self, x):
        if dist.is_available() and dist.is_initialized():
            torch.distributed.nn.functional.all_reduce(
                x, torch.distributed.ReduceOp.AVG
            )
        return x

    @property
    def world_size(self):
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
        return 1

class EppsPulley(UnivariateTest):
    """
    Fast Epps-Pulley two-sample test statistic for univariate distributions.

    This implementation uses numerical integration over the characteristic function
    to compute a goodness-of-fit test statistic. The test compares the empirical
    characteristic function against a standard normal distribution.

    The statistic is computed as:
        T = N * ∫ |φ_empirical(t) - φ_normal(t)|² w(t) dt

    where φ_empirical is the empirical characteristic function, φ_normal is the
    standard normal characteristic function, and w(t) is an integration weight.

    Args:
        t_max (float, optional): Maximum integration point for linear spacing methods.
            Only used for 'trapezoid' and 'simpson' integration. Default: 3.
        n_points (int, optional): Number of integration points. Must be odd for
            'simpson' integration. For 'gauss-hermite', this determines the number
            of positive nodes. Default: 17.
        integration (str, optional): Integration method to use. One of:
            - 'trapezoid': Trapezoidal rule with linear spacing over [0, t_max]
            Default: 'trapezoid'.

    Attributes:
        t (torch.Tensor): Integration points (positive half, including 0).
        weights (torch.Tensor): Precomputed integration weights incorporating
            symmetry and φ(t) = exp(-t²/2).
        phi (torch.Tensor): Precomputed φ(t) = exp(-t²/2) values.
        integration (str): Selected integration method.
        n_points (int): Number of integration points.

    Notes:
        - The implementation exploits symmetry: only t ≥ 0 are computed, and
          contributions from -t are implicitly added via doubled weights.
        - For 'gauss-hermite', nodes and weights are adapted from the standard
          Gauss-Hermite quadrature to integrate against exp(-t²).
        - Supports distributed training via all_reduce operations.
    """

    def __init__(
        self, t_max: float = 5, n_points: int = 17, integration: str = "trapezoid"
    ):
        super().__init__()
        assert n_points % 2 == 1
        self.integration = integration
        self.n_points = n_points
        # Precompute phi

        # Linearly spaced positive points (including 0)
        t = torch.linspace(0, t_max, n_points, dtype=torch.float32)
        self.register_buffer("t", t)
        dt = t_max / (n_points - 1)
        weights = torch.full((n_points,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt  # Half-weight at t=0
        self.register_buffer("phi", self.t.square().mul_(0.5).neg_().exp_())
        self.register_buffer("weights", weights * self.phi)

    def forward(self, x):
        N = x.size(-2)
        # Compute cos/sin only for t >= 0
        x_t = x.unsqueeze(-1) * self.t  # (*, N, K, n_points)
        cos_vals = torch.cos(x_t)
        sin_vals = torch.sin(x_t)

        # Mean across batch
        cos_mean = cos_vals.mean(-3)  # (*, n_points)
        sin_mean = sin_vals.mean(-3)  # (*, n_points)

        # DDP reduction
        cos_mean = all_reduce(cos_mean)
        sin_mean = all_reduce(sin_mean)

        # Compute error (symmetry already in weights)
        err = (cos_mean - self.phi).square() + sin_mean.square()

        # Weighted integration
        return (err @ self.weights) * N * self.world_size
    

