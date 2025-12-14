import logging
import os
import time
from pathlib import Path
import wandb

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def normalize_for_vis(img_np):
    """
    Normalizes a satellite image (H, W, 3) numpy array for visualization
    by clipping to the 2nd and 98th percentiles.
    """
    p2 = np.percentile(img_np, 2)
    p98 = np.percentile(img_np, 98)
    img_norm = np.clip(img_np, p2, p98)
    # Scale to 0-255
    img_norm = (img_norm - p2) / (p98 - p2 + 1e-6)
    img_norm = (img_norm * 255).astype(np.uint8)
    return img_norm

def map_labels_to_colors(label_map, color_map, ignore_index=-1):
    """
    Maps a (H, W) numpy array of class labels to a (H, W, 3) RGB image.
    Pixels with the ignore_index are mapped to black.
    """
    # Create an RGB image, default to black
    rgb_image = np.zeros((label_map.shape[0], label_map.shape[1], 3), dtype=np.uint8)
    
    # Find valid (non-ignored) pixels
    valid_mask = (label_map != ignore_index)
    
    # Get the labels for valid pixels
    valid_labels = label_map[valid_mask]
    
    # Ensure valid_labels are within the color_map range
    valid_labels = np.clip(valid_labels, 0, len(color_map) - 1)
    
    # Map valid labels to colors
    rgb_image[valid_mask] = color_map[valid_labels]
    
    return rgb_image

def save_confusion_matrix(confusion_matrix, class_labels, save_path, title=''):
    """
    Calculates the row-normalized confusion matrix and saves it as a PNG file.
    """
    # Normalize the confusion matrix (rows sum to 1)
    # Add 1e-6 to avoid division by zero for classes with no samples
    cm_normalized = confusion_matrix.astype('float') / (confusion_matrix.sum(axis=1)[:, np.newaxis] + 1e-6)
    
    plt.figure(figsize=(14, 12))  # Increased size for better label readability
    plt.imshow(cm_normalized, cmap=plt.cm.Blues, vmin=0, vmax=1)

    thresh = cm_normalized.max() / 1.7
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            plt.text(j, i, f'{cm_normalized[i, j]:.2f}',  
                     horizontalalignment="center",
                     fontsize=9,
                     color="white" if cm_normalized[i, j] > thresh else "black")
            
    plt.xticks(range(len(class_labels)), class_labels, rotation=45, ha="right", fontsize=10)
    plt.yticks(range(len(class_labels)), class_labels, fontsize=10)
    
    plt.xlabel("Predicted label", fontsize=12)
    plt.ylabel("True label", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)  # Save with high resolution
    plt.close()  # Close the figure to free up memory


class Evaluator:
    """
    Evaluator class for evaluating the models.
    Attributes:
        val_loader (DataLoader): DataLoader for the validation dataset.
        exp_dir (str | Path): Directory for experiment outputs.
        device (torch.device): Device to run the evaluation on (e.g., CPU or GPU).
        use_wandb (bool): Flag to indicate if Weights and Biases (wandb) is used for logging.
        logger (logging.Logger): Logger for logging information.
        classes (list): List of class names in the dataset.
        split (str): Dataset split (e.g., 'train', 'val').
        ignore_index (int): Index to ignore in the dataset.
        num_classes (int): Number of classes in the dataset.
        max_name_len (int): Maximum length of class names.
        wandb (module): Weights and Biases module for logging (if use_wandb is True).
    Methods:
        __init__(val_loader: DataLoader, exp_dir: str | Path, device: torch.device, use_wandb: bool) -> None:
            Initializes the Evaluator with the given parameters.
        evaluate(model: torch.nn.Module, model_name: str, model_ckpt_path: str | Path | None = None) -> None:
            Evaluates the given model. This method should be implemented by subclasses.
        __call__(model: torch.nn.Module) -> None:
            Calls the evaluator on the given model.
        compute_metrics() -> None:
            Computes evaluation metrics. This method should be implemented by subclasses.
        log_metrics(metrics: dict) -> None:
            Logs the computed metrics. This method should be implemented by subclasses.
    """

    def __init__(
            self,
            val_loader: DataLoader,
            distribution: list,
            exp_dir: str | Path,
            device: torch.device,
            use_wandb: bool = False,
            dataset_name: str = 'pastis'
    ) -> None:
        self.rank = int(os.environ["RANK"])
        self.val_loader = val_loader
        self.logger = logging.getLogger()
        self.exp_dir = exp_dir
        self.device = device
        self.classes = self.val_loader.dataset.classes
        self.split = self.val_loader.dataset.split
        self.ignore_index = self.val_loader.dataset.ignore_index
        self.num_classes = len(self.classes)
        self.max_name_len = max([len(name) for name in self.classes])
        self.dataset_name = dataset_name

        self.use_wandb = use_wandb

        self.class_colors = np.array([
            [0, 0, 0],         # Background (Fondo)
            [255, 105, 180],   # GUAYABO (Rosa Chicle)
            [139, 69, 19],     # HIGUERA (Marrón Tierra)
            [255, 255, 0],     # LIMA (Amarillo Brillante)
            [255, 165, 0],     # MANGO (Naranja)
            [255, 140, 0],     # NARANJO (Naranja Oscuro)
            [205, 133, 63],    # NISPERO (Bronceado)
            [255, 192, 203],   # POMELO (Rosa Claro)
            [255, 175, 0],     # TANGELO (Naranja Medio)
            [240, 255, 240],   # ALMENDRO (Blanco Rocío)
            [128, 0, 128],     # CHIRIMOYO (Púrpura)
            [255, 0, 0],       # GRANADO (Rojo Puro)
            [255, 222, 173],   # JOJOBA (Melocotón)
            [154, 205, 50],    # LIMONERO (Verde Limón)
            [255, 153, 51],    # MANDARINO (Mandarina)
            [184, 134, 11],    # MEMBRILLO (Oro Oscuro)
            [255, 99, 71],     # NECTARINO (Rojo Coral)
            [160, 82, 45],     # NOGAL (Siena)
            [128, 128, 0],     # OLIVO (Verde Oliva)
            [0, 100, 0],       # PALTO (Verde Oscuro)
            [255, 20, 147],    # TUNA (Rosa Oscuro)
            [138, 43, 226],    # VID DE MESA (Azul Violeta)
            [70, 130, 180],    # ARANDANO AMERICANO (Azul Acero)
            [220, 20, 60],     # CEREZO (Carmesí)
            [173, 216, 230],   # CIRUELO JAPONES (Azul Claro)
            [244, 164, 96],    # DAMASCO (Arena)
            [210, 105, 30],    # DURAZNERO CONSUMO FRESCO (Chocolate)
            [255, 215, 0],     # DURAZNERO TIPO CONSERVERO (Oro)
            [199, 21, 133],    # FRAMBUESA (Rosa Oscuro)
            [255, 69, 0],      # PAPAYO (Rojo Anaranjado)
            [189, 183, 107],   # PECANA (Caqui Oscuro)
            [0, 255, 0],       # PERAL (Verde Brillante)
            [148, 0, 211],     # CAQUI (Violeta Oscuro)
            [100, 149, 237],   # CIRUELO EUROPEO (Azul Maíz)
            [255, 248, 220],   # FEIJOA (Blanco Crema)
            [0, 255, 255],     # KIWI (Cian)
            [218, 165, 32],    # LUCUMO (Dorado)
            [255, 0, 100],     # MANZANO ROJO (Rojo Vivo)
            [178, 34, 34],     # NUEZ DE MACADAMIA (Ladrillo)
            [64, 224, 208],    # PISTACHO (Turquesa)
            [0, 128, 0],       # HARDY KIWI O BABY KIWI (Verde Medio)
            [128, 0, 0],       # MORAS CULTIVADAS E HIBRIDOS (Marrón Oscuro)
            [0, 0, 139],       # PALMA (Azul Oscuro)
            [192, 192, 192],   # PLUOTS (Plata)
            [165, 42, 42],     # CASTAÑO (Marrón Rojizo)
            [255, 0, 255],     # GROSELLA (Fucsia)
            [176, 196, 222],   # GUINDO AGRIO (Azul Pizarra)
            [0, 200, 0],       # MAQUI (Verde Bosque)
            [255, 0, 200],     # MURTILLA (Rosa Cálido)
            [255, 10, 50],     # ZARZAPARRILLA ROJA (Rojo Carmesí)
            [160, 200, 100],   # MOSQUETA (Verde Salvia)
            [100, 0, 200],     # MICHAY (Índigo)
            [200, 100, 0],     # CRANBERRY (Marrón Anaranjado)
            [255, 255, 100]    # MARACUYA (Amarillo Suave)
        ], dtype=np.uint8)

        self.class_labels = [
    "Background",
    "GUAYABO",
    "HIGUERA",
    "LIMA",
    "MANGO",
    "NARANJO",
    "NISPERO",
    "POMELO",
    "TANGELO",
    "ALMENDRO",
    "CHIRIMOYO",
    "GRANADO",
    "JOJOBA",
    "LIMONERO",
    "MANDARINO",
    "MEMBRILLO",
    "NECTARINO",
    "NOGAL",
    "OLIVO",
    "PALTO",
    "TUNA",
    "VID DE MESA",
    "ARANDANO AMERICANO",
    "CEREZO",
    "CIRUELO JAPONES",
    "DAMASCO",
    "DURAZNERO CONSUMO FRESCO",
    "DURAZNERO TIPO CONSERVERO",
    "FRAMBUESA",
    "PAPAYO",
    "PECANA",
    "PERAL",
    "CAQUI",
    "CIRUELO EUROPEO",
    "FEIJOA",
    "KIWI",
    "LUCUMO",
    "MANZANO ROJO",
    "NUEZ DE MACADAMIA",
    "PISTACHO",
    "HARDY KIWI O BABY KIWI",
    "MORAS CULTIVADAS E HIBRIDOS",
    "PALMA",
    "PLUOTS",
    "CASTANO",
    "GROSELLA",
    "GUINDO AGRIO",
    "MAQUI",
    "MURTILLA",
    "ZARZAPARRILLA ROJA",
    "MOSQUETA",
    "MICHAY",
    "CRANBERRY",
    "MARACUYA"
]

        priors = torch.tensor(distribution, dtype=torch.float32)
        self.log_priors = torch.log(priors).to(self.device)

    def evaluate(
            self,
            model: torch.nn.Module,
            model_name: str,
            model_ckpt_path: str | Path | None = None,
    ) -> None:
        raise NotImplementedError

    def __call__(self, model):
        pass

    def compute_metrics(self):
        pass

    def log_metrics(self, metrics):
        pass

class SegEvaluator(Evaluator):
    """
    SegEvaluator is a class for evaluating segmentation models. It extends the Evaluator class and provides methods
    to evaluate a model, compute metrics, and log the results.
    Attributes:
        val_loader (DataLoader): DataLoader for the validation dataset.
        exp_dir (str | Path): Directory for saving experiment results.
        device (torch.device): Device to run the evaluation on.
        use_wandb (bool): Flag to indicate whether to use Weights and Biases for logging.
    Methods:
        evaluate(model, model_name='model', model_ckpt_path=None):
            Evaluates the given model on the validation dataset and computes metrics.
        __call__(model, model_name, model_ckpt_path=None):
            Calls the evaluate method. This allows the object to be used as a function.
        compute_metrics(confusion_matrix):
            Computes various metrics such as IoU, precision, recall, F1-score, mean IoU, mean F1-score, and mean accuracy
            from the given confusion matrix.
        log_metrics(metrics):
            Logs the computed metrics. If use_wandb is True, logs the metrics to Weights and Biases.
    """

    def __init__(
            self,
            val_loader: DataLoader,
            distribution: list,
            exp_dir: str | Path,
            device: torch.device,
            use_wandb: bool = False,
            dataset_name: str = ""
    ):
        super().__init__(val_loader, distribution, exp_dir, device, use_wandb, dataset_name)

    def reshape_transform(self, tensor, height=15, width=15):
        # Reshape (batch, seq_len, embed_dim) -> (batch, embed_dim, height, width)
        result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))

        # Bring the channels to the first dimension,
        # like in CNNs.
        result = result.transpose(2, 3).transpose(1, 2)
        return result

    @torch.no_grad()
    def evaluate(self, model, model_name='model', model_ckpt_path=None, logit_compensation=False):
        t = time.time()

        if model_ckpt_path is not None:
            model_dict = torch.load(model_ckpt_path, map_location=self.device, weights_only=False)
            model_name = os.path.basename(model_ckpt_path).split(".")[0]
            if "model" in model_dict:
                model.module.load_state_dict(model_dict["model"])
            else:
                model.module.load_state_dict(model_dict)

            self.logger.info(f"Loaded {model_name} for evaluation")
        model.eval()

        tag = f"Evaluating {model_name} on {self.split} set"
        confusion_matrix = torch.zeros(
            (self.num_classes, self.num_classes), device=self.device
        )
        
        # --- NEW: Create directory for visual outputs ---
        vis_save_dir = None
        if self.exp_dir is not None:
            vis_save_dir = os.path.join(self.exp_dir, f"{model_name}_visuals")
            os.makedirs(vis_save_dir, exist_ok=True)
            self.logger.info(f"------------- Saving files to {vis_save_dir} ------------------")
        # --------------------------------------------------

        for batch_idx, data in enumerate(tqdm(self.val_loader, desc=tag)):
            image, target = data["image"], data["target"]
            image_tensor = image["optical"].to(self.device) # Full image tensor
            target = target.to(self.device)
            
            # --- NEW: Keep a copy of original target for visualization ---
            original_target = target.clone()
            # -------------------------------------------------------------
            
            logits = model(image_tensor, batch_positions=data["metadata"], return_feats=False)
            
            if logit_compensation: logits += self.log_priors.view(1, -1, 1, 1)
            
            if logits.shape[1] == 1:
                pred = (torch.sigmoid(logits) > 0.5).type(torch.int64).squeeze(dim=1)
            else:
                pred = torch.argmax(logits, dim=1)
                
            # --- NEW: Keep a copy of original prediction for visualization ---
            original_pred = pred.clone()
            # ---------------------------------------------------------------

            valid_mask = target != self.ignore_index
            pred_masked, target_masked = pred[valid_mask], target[valid_mask]

            count = torch.bincount(
                (pred_masked * self.num_classes + target_masked), minlength=self.num_classes ** 2
            )
            confusion_matrix += count.view(self.num_classes, self.num_classes)
            
            # --- NEW: Save overlay images ---
            # Save only if self.exp_dir is specified and batch size is 1
            if vis_save_dir is not None and image_tensor.shape[0] == 1 and batch_idx <= 200 and batch_idx >= 100:
                try:
                    # 1. Get RGB image: (1, 10, T, H, W) -> (H, W, 3) np.uint8
                    # Select middle temporal instance
                    mid_temporal_idx = image_tensor.shape[2] // 2 
                    # Select BGR (channels 0,1,2) and reorder to RGB (2,1,0)
                    rgb_tensor = image_tensor[0, [3,2,1], mid_temporal_idx, :, :] # (3, H, W)
                    # Convert to (H, W, 3) numpy array
                    rgb_np = rgb_tensor.cpu().permute(1, 2, 0).numpy()
                    # Normalize for visualization
                    rgb_np_vis = normalize_for_vis(rgb_np)
                    img_pil = Image.fromarray(rgb_np_vis).convert('RGBA')
                    original_path = os.path.join(vis_save_dir, f"batch_{batch_idx:04d}_original.png")
                    img_pil.save(original_path)

                    # 2. Get Pred and GT masks: (1, H, W) -> (H, W, 3) np.uint8
                    pred_labels = original_pred[0].cpu().numpy()
                    gt_labels = original_target[0].cpu().numpy()
                    
                    pred_mask_rgb = map_labels_to_colors(pred_labels, self.class_colors, self.ignore_index)
                    gt_mask_rgb = map_labels_to_colors(gt_labels, self.class_colors, self.ignore_index)
                    
                    pred_pil = Image.fromarray(pred_mask_rgb).convert('RGBA')
                    gt_pil = Image.fromarray(gt_mask_rgb).convert('RGBA')
                    
                    # 3. Blend and Save
                    overlay_pred = Image.blend(img_pil, pred_pil, alpha=0.5)
                    overlay_gt = Image.blend(img_pil, gt_pil, alpha=0.5)
                    
                    pred_save_path = os.path.join(vis_save_dir, f"batch_{batch_idx:04d}_pred_overlay.png")
                    gt_save_path = os.path.join(vis_save_dir, f"batch_{batch_idx:04d}_gt_overlay.png")
                    
                    overlay_pred.save(pred_save_path)
                    overlay_gt.save(gt_save_path)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to save visualization for batch {batch_idx}: {e}")
            # --- End of visualization saving ---

        torch.distributed.all_reduce(
            confusion_matrix, op=torch.distributed.ReduceOp.SUM
        )
        
        # --- NEW: Save normalized confusion matrix ---
        if self.exp_dir is not None and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
            cm_save_path = os.path.join(self.exp_dir, f"{model_name}_confusion_matrix.png")
            
            # --- Start: Filter out ignore_index from CM and labels ---
            cm_numpy = confusion_matrix.cpu().numpy()
            
            # Create a boolean mask for all indices that are NOT the ignore_index
            valid_indices_mask = np.arange(self.num_classes) != self.ignore_index
            
            # 1. Filter the confusion matrix (select valid rows, then valid columns)
            cm_filtered = cm_numpy[valid_indices_mask][:, valid_indices_mask]
            
            # 2. Filter the class labels
            labels_filtered = [
                label for i, label in enumerate(self.class_labels) 
                if i != self.ignore_index
            ]
            # --- End: Filtering ---

            save_confusion_matrix(
                cm_filtered,      # Pass the filtered matrix
                labels_filtered,  # Pass the filtered labels
                cm_save_path,
                title=f'{model_name} Normalized Confusion Matrix'
            )
        # ---------------------------------------------
        
        metrics = self.compute_metrics(confusion_matrix.cpu())
        self.log_metrics(metrics)

        used_time = time.time() - t

        return metrics, used_time

    @torch.no_grad()
    def __call__(self, model, model_name, model_ckpt_path=None, logit_compensation=False):
        return self.evaluate(model, model_name, model_ckpt_path, logit_compensation)

    def compute_metrics(self, confusion_matrix):
        if self.ignore_index != -1:
            keep = torch.arange(confusion_matrix.size(0)) != self.ignore_index
            confusion_matrix = confusion_matrix[keep][:, keep]
        
        # Calculate IoU for each class
        intersection = torch.diag(confusion_matrix)
        union = confusion_matrix.sum(dim=1) + confusion_matrix.sum(dim=0) - intersection
        iou = (intersection / (union + 1e-6)) * 100

        # Calculate precision and recall for each class
        precision = intersection / (confusion_matrix.sum(dim=0) + 1e-6) * 100
        recall = intersection / (confusion_matrix.sum(dim=1) + 1e-6) * 100

        # Calculate F1-score for each class
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

        # Calculate mean IoU, mean F1-score, and mean Accuracy
        miou = iou.mean().item()
        mf1 = f1.mean().item()
        macc = (intersection.sum() / (confusion_matrix.sum() + 1e-6)).item() * 100

        # Convert metrics to CPU and to Python scalars
        iou = iou.cpu()
        f1 = f1.cpu()
        precision = precision.cpu()
        recall = recall.cpu()

        # Prepare the metrics dictionary
        metrics = {
            "IoU": [iou[i].item() for i in range(confusion_matrix.size(0))],
            "mIoU": miou,
            "F1": [f1[i].item() for i in range(confusion_matrix.size(0))],
            "mF1": mf1,
            "mAcc": macc,
            "Precision": [precision[i].item() for i in range(confusion_matrix.size(0))],
            "Recall": [recall[i].item() for i in range(confusion_matrix.size(0))],
        }

        return metrics

    def log_metrics(self, metrics):
        def format_metric(name, values, mean_value, classes):
            header = f"------- {name} --------\n"
            metric_str = (
                "\n".join(
                    c.ljust(self.max_name_len, " ") + "\t{:>7}".format("%.3f" % num)
                    for c, num in zip(classes, values)
                )
                + "\n"
            )
            mean_str = (
                "-------------------\n"
                + "Mean".ljust(self.max_name_len, " ")
                + "\t{:>7}".format("%.3f" % mean_value)
            )
            return header + metric_str + mean_str

        # Filter out ignored class if necessary
        if self.ignore_index != -1:
            filtered_classes = [c for i, c in enumerate(self.classes) if i != self.ignore_index]
            iou = [v for i, v in enumerate(metrics["IoU"]) if i != self.ignore_index]
            f1 = [v for i, v in enumerate(metrics["F1"]) if i != self.ignore_index]
            precision = [v for i, v in enumerate(metrics["Precision"]) if i != self.ignore_index]
            recall = [v for i, v in enumerate(metrics["Recall"]) if i != self.ignore_index]
        else:
            filtered_classes = self.classes
            iou = metrics["IoU"]
            f1 = metrics["F1"]
            precision = metrics["Precision"]
            recall = metrics["Recall"]

        iou_str = format_metric("IoU", iou, metrics["mIoU"], filtered_classes)
        f1_str = format_metric("F1-score", f1, metrics["mF1"], filtered_classes)

        precision_mean = torch.tensor(precision).mean().item()
        recall_mean = torch.tensor(recall).mean().item()

        precision_str = format_metric("Precision", precision, precision_mean, filtered_classes)
        recall_str = format_metric("Recall", recall, recall_mean, filtered_classes)

        macc_str = f"Mean Accuracy: {metrics['mAcc']:.3f} \n"

        self.logger.info(iou_str)
        self.logger.info(f1_str)
        self.logger.info(precision_str)
        self.logger.info(recall_str)
        self.logger.info(macc_str)

        if self.use_wandb and self.rank == 0:
            wandb.log(
                {
                    f"{self.split}_mIoU": metrics["mIoU"],
                    f"{self.split}_mF1": metrics["mF1"],
                    f"{self.split}_mAcc": metrics["mAcc"],
                    **{
                        f"{self.split}_IoU_{c}": v
                        for c, v in zip(filtered_classes, iou)
                    },
                    **{
                        f"{self.split}_F1_{c}": v
                        for c, v in zip(filtered_classes, f1)
                    },
                    **{
                        f"{self.split}_Precision_{c}": v
                        for c, v in zip(filtered_classes, precision)
                    },
                    **{
                        f"{self.split}_Recall_{c}": v
                        for c, v in zip(filtered_classes, recall)
                    },
                }
            )
