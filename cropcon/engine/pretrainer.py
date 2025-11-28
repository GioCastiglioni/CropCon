import logging
import operator
import os
import pathlib
import time

import torch
import torch.nn as nn

from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from cropcon.utils.logger import RunningAverageMeter, sec_to_hm
from cropcon.utils.utils import LeJEPATransform as ConsistentTransform

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        teacher: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        n_epochs: int,
        exp_dir: pathlib.Path | str,
        device: torch.device,
        precision: str,
        use_wandb: bool,
        ckpt_interval: int,
        eval_interval: int,
        log_interval: int,
    ):
        """Initialize the Trainer.

        Args:
            model (nn.Module): model to train (encoder + decoder).
            train_loader (DataLoader): train data loader.
            criterion (nn.Module): criterion to compute the loss.
            optimizer (Optimizer): optimizer to update the model's parameters.
            lr_scheduler (LRScheduler): lr scheduler to update the learning rate.
            n_epochs (int): number of epochs to train the model.
            exp_dir (pathlib.Path | str): path to the experiment directory.
            device (torch.device): model
            precision (str): precision to train the model (fp32, fp16, bfp16).
            use_wandb (bool): whether to use wandb for logging.
            ckpt_interval (int): interval to save the checkpoint.
            eval_interval (int): interval to evaluate the model.
            log_interval (int): interval to log the training information.
        """
        self.rank = int(os.environ["RANK"])
        self.criterion = criterion
        self.logit_compensation = str(self.criterion) == "LogitCompensation"
        self.model = model
        self.teacher = teacher
        self.teacher.eval()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.batch_per_epoch = len(self.train_loader)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.n_epochs = n_epochs
        self.logger = logging.getLogger()
        self.exp_dir = exp_dir
        self.device = device
        self.use_wandb = use_wandb
        self.ckpt_interval = ckpt_interval
        self.eval_interval = eval_interval
        self.log_interval = log_interval

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
        
        self.transform = ConsistentTransform(h_w=self.model.module.encoder.input_size, degrees=45).to(self.device)

        self.n_classes = self.model.module.num_classes

        self.ema_momentum = 0.996

        self.grid_size = self.criterion.grid_size[0]
        self.n_target_blocks = 4  # Número de BLOQUES target
        self.target_h = 3         # Altura de un bloque target
        self.target_w = 2         # Ancho de un bloque target
        self.context_h = 7        # Altura del bloque de contexto (ej. ~75%)
        self.context_w = 7        # Ancho del bloque de contexto
    
    def train(self) -> None:
        """Train the model for n_epochs then evaluate the model and save the best model."""
        # end_time = time.time()
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
            self.train_one_epoch(epoch)
            if epoch % self.ckpt_interval == 0 and epoch != self.start_epoch: self.save_model(epoch)
            torch.cuda.empty_cache()

        val_loss = self.evaluate(self.n_epochs)
        self.save_best_checkpoint(val_loss, self.n_epochs)

        # save last model
        self.save_model(self.n_epochs, is_final=True)

        torch.cuda.empty_cache()

    def train_one_epoch(self, epoch: int) -> None:
        """Train model for one epoch.

        Args:
            epoch (int): number of the epoch.
        """
        self.model.train()

        end_time = time.time()
        for batch_idx, data in enumerate(self.train_loader):

            image = self.temporal_transform(data["image"]["optical"].to(self.device))
            B, C, T, H, W = image.shape

            mask_token = self.model.module.mask_token.unsqueeze(2).expand(B, -1, T, -1, -1)

            N_PATCHES = self.grid_size**2
            PATCH_SIZE = H // self.grid_size
            
            mask_ctx_1d = torch.zeros(B, N_PATCHES, device=self.device, dtype=torch.bool)
            mask_tgt_1d = torch.zeros(B, N_PATCHES, device=self.device, dtype=torch.bool)

            for b in range(B):
                top_t = torch.randint(0, self.grid_size - self.target_h + 1, (self.n_target_blocks,), device=self.device)
                left_t = torch.randint(0, self.grid_size - self.target_w + 1, (self.n_target_blocks,), device=self.device)

                mask_tgt_2d = torch.zeros(self.grid_size, self.grid_size, device=self.device, dtype=torch.bool)
                
                idx_target_batch = []
                for i in range(self.n_target_blocks):
                    y_coords = torch.arange(top_t[i], top_t[i] + self.target_h, device=self.device).view(-1, 1)
                    x_coords = torch.arange(left_t[i], left_t[i] + self.target_w, device=self.device).view(1, -1)
                    
                    block_indices_2d = (y_coords * self.grid_size + x_coords).flatten()
                    
                    mask_tgt_2d.view(-1)[block_indices_2d] = True
                    idx_target_batch.append(block_indices_2d)
                
                mask_tgt_1d[b] = mask_tgt_2d.flatten()

                top_c = torch.randint(0, self.grid_size - self.context_h + 1, (1,), device=self.device).item()
                left_c = torch.randint(0, self.grid_size - self.context_w + 1, (1,), device=self.device).item()
                
                mask_ctx_2d = torch.zeros(self.grid_size, self.grid_size, device=self.device, dtype=torch.bool)
                mask_ctx_2d[top_c:top_c+self.context_h, left_c:left_c+self.context_w] = True
                
                mask_ctx_final_2d = mask_ctx_2d & (~mask_tgt_2d)
                
                mask_ctx_1d[b] = mask_ctx_final_2d.flatten()
            
            mask_ctx = mask_ctx_1d.view(B, self.grid_size, self.grid_size)
            
            mask_ctx_pixel = mask_ctx.repeat_interleave(PATCH_SIZE, dim=1).repeat_interleave(PATCH_SIZE, dim=2)

            mask_ctx_5d = mask_ctx_pixel.unsqueeze(1).unsqueeze(2).expand_as(image)
            
            image_context = torch.where(mask_ctx_5d, image, mask_token)
            
            image_target = image
            
            self.training_stats["data_time"].update(time.time() - end_time)

            with torch.autocast(
                "cuda", enabled=self.enable_mixed_precision, dtype=self.precision
            ):
                feat_student = self.model.module.forward_features(image_context, batch_positions=data["metadata"])
                feat_student = self.model.module.patch_conv(feat_student)

                with torch.no_grad():
                    feat_teacher = self.teacher.module.forward_features(image_target, batch_positions=data["metadata"])
                    feat_teacher = self.teacher.module.patch_conv(feat_teacher)

                loss_dict = self.compute_loss(feat_student, feat_teacher, mask_ctx_1d, mask_tgt_1d)

                loss = loss_dict["loss"]

                if loss == 0.0:
                    loss = loss + 0.0 * sum(p.sum() for p in self.model.module.parameters())
                
            self.optimizer.zero_grad()

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Rank {self.rank} got infinite/NaN loss at batch {batch_idx} of epoch {epoch}!"
                )

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self._update_teacher_ema(epoch, batch_idx)

            self.training_stats['loss'].update(loss.item())
            if (batch_idx + 1) % self.log_interval == 0:
                self.log(batch_idx + 1, epoch)

            self.lr_scheduler.step()

            if self.use_wandb and self.rank == 0:
                with torch.no_grad():
                    feat_student_std = feat_student.std()
                    feat_teacher_std = feat_teacher.std()

                self.wandb.log(
                    {
                        "train_loss": loss.item(),
                        "train_student_std": feat_student_std.item(),
                        "train_teacher_std": feat_teacher_std.item(),
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                        "train_mse": loss_dict["mse_loss"].item(),
                        "train_var": loss_dict["var_loss"].item(),
                        "train_cov": loss_dict["cov_loss"].item(),
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
            torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
        return

    @torch.no_grad()
    def evaluate(self, epoch: int):
        """Train model for one epoch.

        Args:
            epoch (int): number of the epoch.
        """
        self.model.eval()

        total_epoch_loss = 0.0
        total_student_std = 0.0
        total_teacher_std = 0.0
        mse_epoch_loss = 0.0
        var_epoch_loss = 0.0
        cov_epoch_loss = 0.0
        
        end_time = time.time()
        for batch_idx, data in enumerate(self.val_loader):

            image = data["image"]["optical"].to(self.device)
            B, C, T, H, W = image.shape
            mask_token = self.model.module.mask_token.unsqueeze(2).expand(B, -1, T, -1, -1)

            N_PATCHES = self.grid_size**2
            PATCH_SIZE = H // self.grid_size
            
            mask_ctx_1d = torch.zeros(B, N_PATCHES, device=self.device, dtype=torch.bool)
            mask_tgt_1d = torch.zeros(B, N_PATCHES, device=self.device, dtype=torch.bool)

            for b in range(B):
                top_t = torch.randint(0, self.grid_size - self.target_h + 1, (self.n_target_blocks,), device=self.device)
                left_t = torch.randint(0, self.grid_size - self.target_w + 1, (self.n_target_blocks,), device=self.device)

                mask_tgt_2d = torch.zeros(self.grid_size, self.grid_size, device=self.device, dtype=torch.bool)
                
                idx_target_batch = []
                for i in range(self.n_target_blocks):
                    y_coords = torch.arange(top_t[i], top_t[i] + self.target_h, device=self.device).view(-1, 1)
                    x_coords = torch.arange(left_t[i], left_t[i] + self.target_w, device=self.device).view(1, -1)
                    
                    block_indices_2d = (y_coords * self.grid_size + x_coords).flatten()
                    
                    mask_tgt_2d.view(-1)[block_indices_2d] = True
                    idx_target_batch.append(block_indices_2d)
                
                mask_tgt_1d[b] = mask_tgt_2d.flatten()

                top_c = torch.randint(0, self.grid_size - self.context_h + 1, (1,), device=self.device).item()
                left_c = torch.randint(0, self.grid_size - self.context_w + 1, (1,), device=self.device).item()
                
                mask_ctx_2d = torch.zeros(self.grid_size, self.grid_size, device=self.device, dtype=torch.bool)
                mask_ctx_2d[top_c:top_c+self.context_h, left_c:left_c+self.context_w] = True
                
                mask_ctx_final_2d = mask_ctx_2d & (~mask_tgt_2d)
                
                mask_ctx_1d[b] = mask_ctx_final_2d.flatten()

            mask_ctx = mask_ctx_1d.view(B, self.grid_size, self.grid_size)
            
            mask_ctx_pixel = mask_ctx.repeat_interleave(PATCH_SIZE, dim=1).repeat_interleave(PATCH_SIZE, dim=2)

            mask_ctx_5d = mask_ctx_pixel.unsqueeze(1).unsqueeze(2).expand_as(image)
            
            image_context = torch.where(mask_ctx_5d, image, mask_token)
            image_target = image 

            self.training_stats["data_time"].update(time.time() - end_time)

            with torch.autocast(
                "cuda", enabled=self.enable_mixed_precision, dtype=self.precision
            ):
                feat_student = self.model.module.forward_features(image_context, batch_positions=data["metadata"])
                feat_student = self.model.module.patch_conv(feat_student)

                feat_teacher = self.teacher.module.forward_features(image_target, batch_positions=data["metadata"])
                feat_teacher = self.teacher.module.patch_conv(feat_teacher)

                avg_batch_loss_dict = self.compute_loss(feat_student, feat_teacher, mask_ctx_1d, mask_tgt_1d)

            total_epoch_loss += avg_batch_loss_dict["loss"].item()
            mse_epoch_loss += avg_batch_loss_dict["mse_loss"].item()
            var_epoch_loss += avg_batch_loss_dict["var_loss"].item()
            cov_epoch_loss += avg_batch_loss_dict["cov_loss"].item()
            
            total_student_std += feat_student.std()
            total_teacher_std += feat_teacher.std()
            torch.distributed.barrier(device_ids=[torch.cuda.current_device()])

        final_val_loss = total_epoch_loss / len(self.val_loader)
        final_mse = mse_epoch_loss / len(self.val_loader)
        final_var = var_epoch_loss / len(self.val_loader)
        final_cov = cov_epoch_loss / len(self.val_loader)
        final_student_std = total_student_std / len(self.val_loader)
        final_teacher_std = total_teacher_std / len(self.val_loader)

        if self.use_wandb and self.rank == 0:
            self.wandb.log(
                {
                    "val_loss": final_val_loss,
                    "val_student_std": final_student_std,
                    "val_teacher_std": final_teacher_std,
                    "val_mse": final_mse,
                    "val_var": final_var,
                    "val_cov": final_cov,
                    "epoch": epoch
                },
                step = epoch * len(self.train_loader)
            )
            
        return final_val_loss

    @torch.no_grad()
    def temporal_transform(self, x: torch.Tensor):
        """
        x:     [B, C, T, H, W]
        """
        B, C, Temp, H, W = x.shape

        # Reshape into [B*T, C, H, W]
        x = x.permute(0, 2, 1, 3, 4).reshape(B*Temp, C, H, W)  # → [B*T, C, H, W]

        # Prepare output tensor
        x_out = torch.empty((B,C,Temp,H,W), device=x.device)

        for b in range(B):
            x_b = x[b*Temp:(b+1)*Temp]  # [T, C, H, W]

            sample = self.transform({"image": x_b})

            x_b = sample["image"].permute(1, 0, 2, 3)

            x_out[b] = x_b

        return x_out

    @torch.no_grad()
    def _update_teacher_ema(self, epoch: int, batch_idx: int):
        
        current_step = epoch * self.batch_per_epoch + batch_idx
        total_steps = self.batch_per_epoch * self.n_epochs
        m = self.ema_momentum + (1 - self.ema_momentum) * (current_step / total_steps)

        student_params = dict(self.model.module.named_parameters())
        teacher_params = dict(self.teacher.module.named_parameters())

        for name, student_param in student_params.items():
            if name in teacher_params:
                teacher_param = teacher_params[name]
                teacher_param.data.mul_(m).add_(student_param.data, alpha=1 - m)
        
        student_buffers = dict(self.model.module.named_buffers())
        teacher_buffers = dict(self.teacher.module.named_buffers())

        for name, student_buffer in student_buffers.items():
            if name in teacher_buffers:
                teacher_buffer = teacher_buffers[name]
                teacher_buffer.data.copy_(student_buffer.data)

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
        checkpoint = checkpoint["model"]
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
            eval_metrics (dict[float, list[float]]): metrics computed on the validation set.
            epoch (int): number of the epoch.
        """
        if self.best_metric_comp(loss, self.best_metric):
            self.best_metric = loss
            best_ckpt = self.get_checkpoint(epoch)
            self.save_model(
                epoch, is_best=True, checkpoint=best_ckpt
            )

    def compute_loss(self, feat_v1: torch.Tensor, feat_v2: torch.Tensor, idx1: torch.Tensor, idx2: torch.Tensor) -> torch.Tensor:
        """Compute the loss"""
        return self.criterion(feat_v1, feat_v2, idx1, idx2)

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

