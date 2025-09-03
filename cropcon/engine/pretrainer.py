import copy
import logging
import operator
import os
import pathlib
import time
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms.v2 as T

from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Subset

from cropcon.utils.logger import RunningAverageMeter, sec_to_hm
from cropcon.utils.losses import CropConLoss
from cropcon.utils.utils import ConsistentTransform
from scipy.ndimage import label as lbl
from grokfast import gradfilter_ma, gradfilter_ema

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        projector: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        distribution: list,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        evaluator: torch.nn.Module,
        n_epochs: int,
        exp_dir: pathlib.Path | str,
        device: torch.device,
        precision: str,
        use_wandb: bool,
        ckpt_interval: int,
        eval_interval: int,
        log_interval: int,
        tau: float,
        alpha: float,
        projection_dim: int,
    ):
        """Initialize the Trainer.

        Args:
            model (nn.Module): model to train (encoder + decoder).
            train_loader (DataLoader): train data loader.
            criterion (nn.Module): criterion to compute the loss.
            distribution (list): class distributions.
            optimizer (Optimizer): optimizer to update the model's parameters.
            lr_scheduler (LRScheduler): lr scheduler to update the learning rate.
            evaluator (torch.nn.Module): task evaluator to evaluate the model.
            n_epochs (int): number of epochs to train the model.
            exp_dir (pathlib.Path | str): path to the experiment directory.
            device (torch.device): model
            precision (str): precision to train the model (fp32, fp16, bfp16).
            use_wandb (bool): whether to use wandb for logging.
            ckpt_interval (int): interval to save the checkpoint.
            eval_interval (int): interval to evaluate the model.
            log_interval (int): interval to log the training information.
            tau (float): temperature parameter for SupCon.
            alpha (float): weighting factor for CE and SupCon losses.
        """
        self.rank = int(os.environ["RANK"])
        self.criterion = criterion
        self.logit_compensation = str(self.criterion) == "LogitCompensation"
        self.distribution = distribution
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.batch_per_epoch = len(self.train_loader)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.evaluator = evaluator
        self.n_epochs = n_epochs
        self.logger = logging.getLogger()
        self.exp_dir = exp_dir
        self.device = device
        self.use_wandb = use_wandb
        self.ckpt_interval = ckpt_interval
        self.eval_interval = eval_interval
        self.log_interval = log_interval
        self.projection_dim=projection_dim
        self.grokfast = False

        self.training_stats = {
            name: RunningAverageMeter(length=self.batch_per_epoch)
            for name in ["loss", "data_time", "batch_time"]
        }
        self.training_metrics = {}
        self.best_metric = float("inf")
        self.best_metric_comp = operator.lt
        self.num_classes = self.train_loader.dataset.num_classes

        assert precision in [
            "fp32",
            "fp16",
            "bfp16",
        ], f"Invalid precision {precision}, use 'fp32', 'fp16' or 'bfp16'."
        self.enable_mixed_precision = precision != "fp32"
        self.precision = torch.float16 if (precision == "fp16") else torch.bfloat16
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.enable_mixed_precision)

        self.start_epoch = 0

        if self.use_wandb:
            import wandb

            self.wandb = wandb
        
        self.alpha = alpha

        self.projector = projector
        
        self.transform1 = ConsistentTransform(h_w=self.model.module.encoder.input_size, degrees=45, view=1).to(self.device)
        self.transform2 = ConsistentTransform(h_w=self.model.module.encoder.input_size, degrees=45, view=2).to(self.device)

        self.n_classes = self.model.module.num_classes
    
    def train(self) -> None:
        """Train the model for n_epochs then evaluate the model and save the best model."""
        # end_time = time.time()
        grads=None
        grads_proj=None
        for epoch in range(self.start_epoch, self.n_epochs):
            # train the network for one epoch
            if epoch % self.eval_interval == 0:
                self.logger.info(f"Evaluating epoch {epoch}...")
                val_loss = self.evaluate(epoch)
                self.save_best_checkpoint(val_loss, epoch)
                self.logger.info(f"Evaluation complete.")
                torch.cuda.empty_cache()

            self.logger.info("============ Starting epoch %i ... ============" % epoch)
            # set sampler
            self.t = time.time()
            self.train_loader.sampler.set_epoch(epoch)
            grads, grads_proj = self.train_one_epoch(epoch, grads=grads, grads_proj=grads_proj)
            if epoch % self.ckpt_interval == 0 and epoch != self.start_epoch: self.save_model(epoch)
            torch.cuda.empty_cache()

        val_loss = self.evaluate(self.n_epochs)
        self.save_best_checkpoint(val_loss, self.n_epochs)

        # save last model
        self.save_model(self.n_epochs, is_final=True)

        torch.cuda.empty_cache()

    def train_one_epoch(self, epoch: int, grads=None, grads_proj=None) -> None:
        """Train model for one epoch.

        Args:
            epoch (int): number of the epoch.
        """
        self.model.train()

        end_time = time.time()
        for batch_idx, data in enumerate(self.train_loader):

            image = {"v1": self.temporal_transform(data["image"]["optical"].to(self.device), view=1)}
            image["v2"] = self.temporal_transform(data["image"]["optical"].to(self.device), view=2)

            self.training_stats["data_time"].update(time.time() - end_time)

            with torch.autocast(
                "cuda", enabled=self.enable_mixed_precision, dtype=self.precision
            ):
                
                feat_con, _, _, _ = self.model.module.forward_bottleneck(
                    torch.cat([
                        image["v1"],
                        image["v2"]], dim=0),
                    batch_positions=torch.cat([data["metadata"], data["metadata"]], dim=0)
                )

                feat_con = feat_con.mean(dim=(-2,-1))

                proj = self.projector(feat_con)

                loss = self.compute_loss(
                    proj[:proj.shape[0] // 2],
                    proj[proj.shape[0] // 2:]
                )
                
            self.optimizer.zero_grad()

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Rank {self.rank} got infinite/NaN loss at batch {batch_idx} of epoch {epoch}!"
                )

            self.scaler.scale(loss).backward()
            if self.grokfast:
                self.scaler.unscale_(self.optimizer)
                grads = gradfilter_ema(self.model, grads=grads)
                grads_proj = gradfilter_ema(self.projector, grads=grads_proj)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.training_stats['loss'].update(loss.item())
            if (batch_idx + 1) % self.log_interval == 0:
                self.log(batch_idx + 1, epoch)

            self.lr_scheduler.step()

            if self.use_wandb and self.rank == 0:
                self.wandb.log(
                    {
                        "train_loss": loss.item(),
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                        **{
                            f"train_{k}": v.avg
                            for k, v in self.training_metrics.items()
                        },
                    },
                    step=epoch * len(self.train_loader) + batch_idx,
                )

            self.training_stats["batch_time"].update(time.time() - end_time)
            end_time = time.time()
        return grads, grads_proj

    @torch.no_grad()
    def evaluate(self, epoch: int):
        """Train model for one epoch.

        Args:
            epoch (int): number of the epoch.
        """
        self.model.eval()

        end_time = time.time()
        loss = 0
        for batch_idx, data in enumerate(self.val_loader):

            image = {"v1": self.temporal_transform(data["image"]["optical"].to(self.device), view=1)}
            image["v2"] = self.temporal_transform(data["image"]["optical"].to(self.device), view=2)

            with torch.autocast(
                "cuda", enabled=self.enable_mixed_precision, dtype=self.precision
            ):
                
                feat_con, _, _, _ = self.model.module.forward_bottleneck(
                    torch.cat([
                        image["v1"],
                        image["v2"]], dim=0),
                    batch_positions=torch.cat([data["metadata"], data["metadata"]], dim=0)
                )

                feat_con = feat_con.mean(dim=(-2,-1))

                proj = self.projector(feat_con)

                batch_loss = self.compute_loss(proj[:proj.shape[0] // 2], proj[proj.shape[0] // 2:])

                if batch_idx % self.log_interval == 0: self.logger.info(f"Val batch: {batch_idx+1}/{len(self.val_loader)}")

                loss += batch_loss.item()

        if self.use_wandb and self.rank == 0:
            self.wandb.log(
                {
                    "val_loss": loss/(batch_idx+1),
                    "epoch": epoch
                },
                step = epoch * len(self.train_loader)
            )
        return batch_loss
    
    def temporal_transform(self, x: torch.Tensor, view: int = 1):
        """
        x:     [B, C, T, H, W]
        """
        B, C, Temp, H, W = x.shape

        # Reshape into [B*T, C, H, W]
        x = x.permute(0, 2, 1, 3, 4).reshape(B*Temp, C, H, W)  # → [B*T, C, H, W]

        # Prepare output tensors
        x_out = torch.empty((B,C,Temp,H,W), device=x.device)

        for b in range(B):
            x_b = x[b*Temp:(b+1)*Temp]  # [T, C, H, W]

            sample = self.transform1({"image": x_b}) if view == 1 else self.transform2({"image": x_b})

            x_b = sample["image"].permute(1, 0, 2, 3)

            x_out[b] = x_b

        return x_out
    
    @torch.no_grad()
    def extract_crop_features(self, features: torch.Tensor, gt_masks: torch.Tensor):
        B, D, H, W = features.shape
        device = features.device
        crop_vecs = []
        crop_labels = []

        for b in range(B):
            feat = features[b]  # [D, H, W]
            gt_mask_np = gt_masks[b].cpu().numpy()

            for class_id in np.unique(gt_mask_np):
                if class_id == self.criterion.ignore_index:
                    continue

                class_mask = (gt_mask_np == class_id).astype(np.uint8)
                labeled, num_features = lbl(class_mask)

                for i in range(1, num_features + 1):
                    crop_mask = (labeled == i)
                    if crop_mask.sum() == 0:
                        continue

                    # Convert to torch indices
                    y_idx, x_idx = np.nonzero(crop_mask)
                    y_idx = torch.from_numpy(y_idx).to(device)
                    x_idx = torch.from_numpy(x_idx).to(device)

                    crop_feat = feat[:, y_idx, x_idx]  # [D, N]
                    crop_avg = crop_feat.mean(dim=1)   # [D]
                    crop_vecs.append(crop_avg)
                    crop_labels.append(int(class_id))

        if not crop_vecs:
            return (torch.empty(0, D, device=device),
                    torch.empty(0, dtype=torch.long, device=device))

        crop_features = torch.stack(crop_vecs, dim=0)  # [num_crops, D]
        crop_labels = torch.tensor(crop_labels, dtype=torch.long, device=device)
        return crop_features, crop_labels
    
    def get_checkpoint(self, epoch: int) -> dict[str, dict | int]:
        """Create a checkpoint dictionary, containing references to the pytorch tensors.

        Args:
            epoch (int): number of the epoch.

        Returns:
            dict[str, dict | int]: checkpoint dictionary.
        """
        checkpoint = {
            "model": self.model.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": epoch,
        }
        return checkpoint

    def save_model(
        self,
        epoch: int,
        is_final: bool = False,
        is_best: bool = False,
        checkpoint: dict[str, dict | int] | None = None,
    ):
        """Save the model checkpoint.

        Args:
            epoch (int): number of the epoch.
            is_final (bool, optional): whether is the final checkpoint. Defaults to False.
            is_best (bool, optional): wheter is the best checkpoint. Defaults to False.
            checkpoint (dict[str, dict  |  int] | None, optional): already prepared checkpoint dict. Defaults to None.
        """
        if self.rank != 0:
            torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
            return
        checkpoint = self.get_checkpoint(epoch) if checkpoint is None else checkpoint
        suffix = "_best" if is_best else f"{epoch}_final" if is_final else f"{epoch}"
        checkpoint_path = os.path.join(self.exp_dir, f"checkpoint_{suffix}.pth")
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(
            f"Epoch {epoch} | Training checkpoint saved at {checkpoint_path}"
        )
        torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
        return

    def load_model(self, resume_path: str | pathlib.Path) -> None:
        """Load model from the checkpoint.

        Args:
            resume_path (str | pathlib.Path): path to the checkpoint.
        """
        model_dict = torch.load(resume_path, map_location=self.device, weights_only=False)
        if "model" in model_dict:
            self.model.module.load_state_dict(model_dict["model"])
            self.optimizer.load_state_dict(model_dict["optimizer"])
            self.lr_scheduler.load_state_dict(model_dict["lr_scheduler"])
            self.scaler.load_state_dict(model_dict["scaler"])
            self.start_epoch = model_dict["epoch"] + 1
        else:
            self.model.module.load_state_dict(model_dict)
            self.start_epoch = 0

        self.logger.info(
            f"Loaded model from {resume_path}. Resume training from epoch {self.start_epoch}"
        )

    def save_best_checkpoint(
        self, loss: float, epoch: int
    ) -> None:
        """Update the best checkpoint according to the loss.

        Args:
            eval_metrics (dict[float, list[float]]): metrics computed by the evaluator on the validation set.
            epoch (int): number of the epoch.
        """
        if self.best_metric_comp(loss, self.best_metric):
            self.best_metric = loss
            best_ckpt = self.get_checkpoint(epoch)
            self.save_model(
                epoch, is_best=True, checkpoint=best_ckpt
            )

    def compute_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the loss.

        Args:
            logits (torch.Tensor): logits from the decoder.
            target (torch.Tensor): target tensor.

        Returns:
            torch.Tensor: loss value.
        """
        return self.criterion(logits, target)

    def log(self, batch_idx: int, epoch) -> None:
        """Log the information.

        Args:
            batch_idx (int): number of the batch.
            epoch (_type_): number of the epoch.
        """
        # TO DO: upload to wandb
        left_batch_this_epoch = self.batch_per_epoch - batch_idx
        left_batch_all = (
            self.batch_per_epoch * (self.n_epochs - epoch - 1) + left_batch_this_epoch
        )
        left_time_this_epoch = sec_to_hm(
            left_batch_this_epoch * self.training_stats["batch_time"].avg
        )

        basic_info = (
            "Epoch [{epoch}-{batch_idx}/{len_loader}]\t"
            "ETA [{left_time_this_epoch}]\t"
            "Time [{batch_time.avg:.3f}|{data_time.avg:.3f}]\t"
            "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
            "lr {lr:.3e}".format(
                epoch=epoch,
                len_loader=len(self.train_loader),
                batch_idx=batch_idx,
                left_time_this_epoch=left_time_this_epoch,
                batch_time=self.training_stats["batch_time"],
                data_time=self.training_stats["data_time"],
                loss=self.training_stats["loss"],
                lr=self.optimizer.param_groups[0]["lr"],
            )
        )

        self.logger.info(basic_info)

