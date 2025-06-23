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
from cropcon.utils.losses import SupContrastiveLoss, BCLLoss
from cropcon.utils.utils import RandomChannelDropout, ConsistentTransform

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        projector: nn.Module,
        prototype_projector: nn.Module,
        train_loader: DataLoader,
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
        best_metric_key: str,
        tau: float,
        alpha: float,
        projection_dim: int = 128
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
            best_metric_key (str): metric that determines best checkpoints.
            tau (float): temperature parameter for SupCon.
            alpha (float): weighting factor for CE and SupCon losses.
        """
        self.rank = int(os.environ["RANK"])
        self.criterion = criterion
        self.logit_compensation = str(self.criterion) == "LogitCompensation"
        self.distribution = distribution
        self.model = model
        self.train_loader = train_loader
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
        self.best_metric_key = best_metric_key
        self.projection_dim=projection_dim

        self.training_stats = {
            name: RunningAverageMeter(length=self.batch_per_epoch)
            for name in ["loss", "data_time", "batch_time", "eval_time"]
        }
        self.training_metrics = {}
        self.best_metric_comp = operator.gt
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

        self.channel_drop = T.Compose([RandomChannelDropout(p=0.5, max_drop=5)])
        
        self.alpha = alpha

        self.contrastive = BCLLoss(tau=tau, ignore_index=self.criterion.ignore_index)

        self.projector = projector
        
        self.prototype_projector = prototype_projector
        
        self.transform = ConsistentTransform(degrees=45, p=0.5).to(self.device)

        self.n_classes = self.model.module.num_classes
        
        self.prototypes = torch.zeros((self.n_classes, self.projection_dim), device=self.device).requires_grad_(False)
        self.prototype_initialized = torch.zeros(self.n_classes, dtype=torch.bool, device=self.device)

        self.momentum_ema = lambda epoch: min(0.9, 0.7 + 0.04 * epoch)
    
    def train(self) -> None:
        """Train the model for n_epochs then evaluate the model and save the best model."""
        # end_time = time.time()
        for epoch in range(self.start_epoch, self.n_epochs):
            # train the network for one epoch
            if epoch % self.eval_interval == 0:
                metrics, used_time = self.evaluator(self.model, f"epoch {epoch}", logit_compensation=self.logit_compensation)
                self.training_stats["eval_time"].update(used_time)
                self.save_best_checkpoint(metrics, epoch)
                del metrics
                del used_time
                torch.cuda.empty_cache()

            self.logger.info("============ Starting epoch %i ... ============" % epoch)
            # set sampler
            self.t = time.time()
            self.train_loader.sampler.set_epoch(epoch)
            self.train_one_epoch(epoch)
            if epoch % self.ckpt_interval == 0 and epoch != self.start_epoch: self.save_model(epoch)
            torch.cuda.empty_cache()

        metrics, used_time = self.evaluator(self.model, "final model")
        self.training_stats["eval_time"].update(used_time)
        self.save_best_checkpoint(metrics, self.n_epochs)

        del metrics
        del used_time
        torch.cuda.empty_cache()

    def extract_classwise_representations(self, feature_maps, mask):
        B, C, Hf, Wf = feature_maps.shape
        features = []
        targets = []

        for b in range(B):
            fmap = feature_maps[b]           # [C, Hf, Wf]
            class_ids = torch.unique(mask[b])

            for cls_id in class_ids:
                cls_mask = (mask[b] == cls_id)  # [Hf, Wf]
                if cls_mask.sum() == 0:
                    continue  # just in case

                cls_mask_flat = cls_mask.view(1, -1)                   # [1, Hf*Wf]
                fmap_flat = fmap.view(C, -1)                           # [C, Hf*Wf]
                selected_features = fmap_flat[:, cls_mask_flat[0]]    # [C, N_pixels]
                pooled_vector = selected_features.mean(dim=1)         # [C]
                
                features.append(pooled_vector)
                targets.append(cls_id.item())

        feature_tensor = torch.stack(features, dim=0)   # [P, C]
        target_tensor = torch.tensor(targets, device=feature_maps.device)   # [P]

        return feature_tensor, target_tensor
    
    def temporal_transform(self, x: torch.Tensor, mask: torch.Tensor):
        """
        x:     [B, C, T, H, W]
        mask:  [B, H, W]
        """
        B, C, Temp, H, W = x.shape

        # Reshape into [B*T, C, H, W]
        x = x.permute(0, 2, 1, 3, 4).reshape(B*Temp, C, H, W)  # → [B*T, C, H, W]

        # Prepare output tensors
        x_out = torch.empty((B,C,Temp,H,W), device=x.device)
        mask_out = torch.empty((B,H,W), device=x.device)

        for b in range(B):
            x_b = x[b*Temp:(b+1)*Temp]  # [T, C, H, W]

            m_b = mask[b].expand(Temp, H, W).unsqueeze(1)
            sample = self.transform({"image": x_b, "mask": m_b})

            x_b = sample["image"].permute(1, 0, 2, 3)
            m_b = sample["mask"][0].squeeze()

            x_out[b] = x_b
            mask_out[b] = m_b

        return x_out, mask_out

    def compute_class_prototypes(self, feat_con1, target_con1):
        D = feat_con1.size(1)

        # Inicializar acumuladores
        sums = torch.zeros(self.n_classes, D, device=self.device)
        counts = torch.zeros(self.n_classes, device=self.device)

        # Sumar todos los vectores por clase
        for c in torch.unique(target_con1).long():
            mask = (target_con1 == c)
            sums[c] = feat_con1[mask].sum(dim=0)
            counts[c] = mask.sum()

        # Evitar división por cero
        counts = counts.clamp(min=1)

        # Obtener promedio
        prototypes = sums / counts.unsqueeze(1)
        
        return prototypes.detach()
    
    @torch.no_grad()
    def update_prototypes_distributed(self, new_prototype, target_con1, epoch):
        world_size = torch.distributed.get_world_size()

        cls_list = torch.unique(target_con1).long()
        proto_dim = new_prototype.size(1)

        # Prepare containers
        global_updates = torch.zeros_like(self.prototypes)  # (C, D)
        counts = torch.zeros(self.n_classes, device=self.device)

        for cls in cls_list:
            global_updates[cls] = new_prototype[cls].detach()
            counts[cls] = 1.0

        # Sum across all GPUs
        torch.distributed.all_reduce(global_updates, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)

        # Compute global averages for the classes that appeared on any GPU
        valid = counts > 0
        global_updates[valid] /= counts[valid].unsqueeze(1)

        # Update prototypes with EMA
        for cls in torch.where(valid)[0]:
            cls = cls.item()
            if not self.prototype_initialized[cls]:
                self.prototypes[cls] = F.normalize(global_updates[cls], dim=0)
                self.prototype_initialized[cls] = True
            else:
                updated = (
                    self.momentum_ema(epoch) * self.prototypes[cls]
                    + (1 - self.momentum_ema(epoch)) * global_updates[cls]
                )
                self.prototypes[cls] = F.normalize(updated, dim=0)

        # Sync initialized flags across GPUs
        proto_flags = self.prototype_initialized.float()
        torch.distributed.all_reduce(proto_flags, op=torch.distributed.ReduceOp.MAX)
        self.prototype_initialized = proto_flags.bool()
    
    def train_one_epoch(self, epoch: int) -> None:
        """Train model for one epoch.

        Args:
            epoch (int): number of the epoch.
        """
        self.model.train()

        end_time = time.time()
        for batch_idx, data in enumerate(self.train_loader):

            image, target = data["image"], data["target"]
            image = {"v1": image["optical"].to(self.device)}
            target = target.to(self.device)

            with torch.no_grad():
                image["v2"], target_transformed2 = self.temporal_transform(image["v1"].detach().clone(), target.clone())
                image["v3"], target_transformed3 = self.temporal_transform(image["v1"].detach().clone(), target.clone())

            self.training_stats["data_time"].update(time.time() - end_time)

            with torch.autocast(
                "cuda", enabled=self.enable_mixed_precision, dtype=self.precision
            ):
                logits, feat_v1  = self.model(image["v1"],
                                              batch_positions=data["metadata"],
                                              return_feats=True)
                loss_ce = self.compute_loss(logits, target)
                if self.alpha > 0.0: 
                    feat_con1, target_con1 = self.extract_classwise_representations(
                        feat_v1,
                        target.unsqueeze(1).float()
                        )
                    
                    new_prototype = self.compute_class_prototypes(feat_con1, target_con1)
                    new_prototype = F.normalize(self.prototype_projector(new_prototype), dim=1)

                    self.update_prototypes_distributed(new_prototype, target_con1, epoch)

                    feats = self.model.module.forward_features(torch.cat((image["v2"],image["v3"]), dim=0),
                                                batch_positions=data["metadata"].repeat(2, 1))

                    feat_v2 = feats[:image["v2"].shape[0]]
                    feat_con2, target_con2 = self.extract_classwise_representations(
                        feat_v2,
                        target_transformed2.unsqueeze(1).float()
                        )

                    feat_v3  = feats[image["v2"].shape[0]:]
                    feat_con3, target_con3 = self.extract_classwise_representations(
                        feat_v3,
                        target_transformed3.unsqueeze(1).float()
                        )

                    projs = self.projector(torch.cat((feat_con2,feat_con3), dim=0))

                    proj2 = projs[:image["v2"].shape[0]]
                    proj3 = projs[image["v2"].shape[0]:]

                    loss_contrastive = self.contrastive(self.prototypes, proj2, target_con2, proj3, target_con3)

                else: loss_contrastive = 0.0
                loss = loss_ce + self.alpha*loss_contrastive

            self.optimizer.zero_grad()

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Rank {self.rank} got infinite/NaN loss at batch {batch_idx} of epoch {epoch}!"
                )

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.training_stats['loss'].update(loss.item())
            with torch.no_grad():
                self.compute_logging_metrics(logits, target)
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
            torch.distributed.barrier()
            return
        checkpoint = self.get_checkpoint(epoch) if checkpoint is None else checkpoint
        suffix = "_best" if is_best else f"{epoch}_final" if is_final else f"{epoch}"
        checkpoint_path = os.path.join(self.exp_dir, f"checkpoint_{suffix}.pth")
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(
            f"Epoch {epoch} | Training checkpoint saved at {checkpoint_path}"
        )
        torch.distributed.barrier()
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

    def compute_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the loss.

        Args:
            logits (torch.Tensor): logits from the model.
            target (torch.Tensor): target tensor.

        Raises:
            NotImplementedError: raise if the method is not implemented.

        Returns:
            torch.Tensor: loss value.
        """
        raise NotImplementedError

    def save_best_checkpoint(
        self, eval_metrics: dict[float, list[float]], epoch: int
    ) -> None:
        """Update the best checkpoint according to the evaluation metrics.

        Args:
            eval_metrics (dict[float, list[float]]): metrics computed by the evaluator on the validation set.
            epoch (int): number of the epoch.
        """
        curr_metric = eval_metrics[self.best_metric_key]
        if isinstance(curr_metric, list):
            curr_metric = curr_metric[0] if self.num_classes == 1 else np.mean(curr_metric)
        if self.best_metric_comp(curr_metric, self.best_metric):
            self.best_metric = curr_metric
            best_ckpt = self.get_checkpoint(epoch)
            best_ckpt["mIoU"] = eval_metrics["mIoU"]
            best_ckpt["mF1"] = eval_metrics["mF1"]
            best_ckpt["mAcc"] = eval_metrics["mAcc"]
            self.save_model(
                epoch, is_best=True, checkpoint=best_ckpt
            )

    @torch.no_grad()
    def compute_logging_metrics(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> dict[float, list[float]]:
        """Compute logging metrics.

        Args:
            logits (torch.Tensor): logits output by the decoder.
            target (torch.Tensor): target tensor.

        Raises:
            NotImplementedError: raise if the method is not implemented.

        Returns:
            dict[float, list[float]]: logging metrics.
        """
        raise NotImplementedError

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
        left_eval_times = ((self.n_epochs - 0.5) // self.eval_interval + 2
                           - self.training_stats["eval_time"].count)
        left_time_this_epoch = sec_to_hm(
            left_batch_this_epoch * self.training_stats["batch_time"].avg
        )
        left_time_all = sec_to_hm(
            left_batch_all * self.training_stats["batch_time"].avg
            + left_eval_times * self.training_stats["eval_time"].avg
        )

        basic_info = (
            "Epoch [{epoch}-{batch_idx}/{len_loader}]\t"
            "ETA [{left_time_all}|{left_time_this_epoch}]\t"
            "Time [{batch_time.avg:.3f}|{data_time.avg:.3f}]\t"
            "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
            "lr {lr:.3e}".format(
                epoch=epoch,
                len_loader=len(self.train_loader),
                batch_idx=batch_idx,
                left_time_this_epoch=left_time_this_epoch,
                left_time_all=left_time_all,
                batch_time=self.training_stats["batch_time"],
                data_time=self.training_stats["data_time"],
                loss=self.training_stats["loss"],
                lr=self.optimizer.param_groups[0]["lr"],
            )
        )

        metrics_info = [
            "{} {:>7} ({:>7})".format(k, "%.3f" % v.val, "%.3f" % v.avg)
            for k, v in self.training_metrics.items()
        ]
        metrics_info = "\n Training metrics: " + "\t".join(metrics_info)
        # extra_metrics_info = self.extra_info_template.format(**self.extra_info)
        log_info = basic_info + metrics_info
        self.logger.info(log_info)

    def reset_stats(self) -> None:
        """Reset the training stats and metrics."""
        for v in self.training_stats.values():
            v.reset()
        for v in self.training_metrics.values():
            v.reset()


class SegTrainer(Trainer):
    def __init__(
        self,
        model: nn.Module,
        projector: nn.Module,
        prototype_projector: nn.Module,
        train_loader: DataLoader,
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
        best_metric_key: str,
        tau: float,
        alpha: float,
        projection_dim: int = 128,
    ):
        """Initialize the Trainer for segmentation task.
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
            best_metric_key (str): metric that determines best checkpoints.
        """
        super().__init__(
            model=model,
            projector=projector,
            prototype_projector=prototype_projector,
            projection_dim=projection_dim,
            train_loader=train_loader,
            criterion=criterion,
            distribution=distribution,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            evaluator=evaluator,
            n_epochs=n_epochs,
            exp_dir=exp_dir,
            device=device,
            precision=precision,
            use_wandb=use_wandb,
            ckpt_interval=ckpt_interval,
            eval_interval=eval_interval,
            log_interval=log_interval,
            best_metric_key=best_metric_key,
            tau=tau,
            alpha=alpha
        )

        self.training_metrics = {
            name: RunningAverageMeter(length=100) for name in ["Acc", "mAcc", "mIoU"]
        }
        self.best_metric = float("-inf")
        self.best_metric_comp = operator.gt

    def compute_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the loss.

        Args:
            logits (torch.Tensor): logits from the decoder.
            target (torch.Tensor): target tensor.

        Returns:
            torch.Tensor: loss value.
        """
        return self.criterion(logits, target)

    @torch.no_grad()
    def compute_logging_metrics(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> None:
        """Compute logging metrics.

        Args:
            logits (torch.Tensor): loggits from the decoder.
            target (torch.Tensor): target tensor.
        """
        # logits = F.interpolate(logits, size=target.shape[1:], mode='bilinear')
        num_classes = logits.shape[1]
        if num_classes == 1:
            pred = (torch.sigmoid(logits) > 0.5).type(torch.int64)
        else:
            pred = torch.argmax(logits, dim=1, keepdim=True)
        target = target.unsqueeze(1)
        ignore_mask = target == self.train_loader.dataset.ignore_index
        target[ignore_mask] = 0
        ignore_mask = ignore_mask.expand(
            -1, num_classes if num_classes > 1 else 2, -1, -1
        )

        dims = list(logits.shape)
        if num_classes == 1:
            dims[1] = 2
        binary_pred = torch.zeros(dims, dtype=bool, device=self.device)
        binary_target = torch.zeros(dims, dtype=bool, device=self.device)
        binary_pred.scatter_(dim=1, index=pred, src=torch.ones_like(binary_pred))
        binary_target.scatter_(dim=1, index=target, src=torch.ones_like(binary_target))
        binary_pred[ignore_mask] = 0
        binary_target[ignore_mask] = 0

        intersection = torch.logical_and(binary_pred, binary_target)
        union = torch.logical_or(binary_pred, binary_target)

        acc = intersection.sum() / binary_target.sum() * 100
        macc = (
            torch.nanmean(
                intersection.sum(dim=(0, 2, 3)) / binary_target.sum(dim=(0, 2, 3))
            )
            * 100
        )
        miou = (
            torch.nanmean(intersection.sum(dim=(0, 2, 3)) / union.sum(dim=(0, 2, 3)))
            * 100
        )

        self.training_metrics["Acc"].update(acc.item())
        self.training_metrics["mAcc"].update(macc.item())
        self.training_metrics["mIoU"].update(miou.item())

