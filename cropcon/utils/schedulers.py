import torch
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer


def MultiStepLR(
    optimizer: Optimizer, total_iters: int, lr_milestones: list[float]
) -> LRScheduler:
    
    """
    This module provides a utility function for creating a multi-step learning rate scheduler.

    Functions:
        MultiStepLR(optimizer: Optimizer, total_iters: int, lr_milestones: list[float]) -> LRScheduler
            Creates a MultiStepLR scheduler with specified milestones and decay factor.

    Args:
        optimizer (Optimizer): The optimizer for which to schedule the learning rate.
        total_iters (int): The total number of iterations for training.
        lr_milestones (list[float]): A list of fractions representing the milestones at which the learning rate will be decayed.

    Returns:
        LRScheduler: A PyTorch learning rate scheduler that decays the learning rate at specified milestones.
    """

    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer, [int(total_iters * r) for r in lr_milestones], gamma=0.1
    )


def CosineAnnealingLR(
    optimizer: Optimizer, total_iters: int, lr_milestones: list[float]
) -> LRScheduler:
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(total_iters*lr_milestones[0]))

def CosineAnnealingWarmRestarts(
    optimizer: Optimizer, total_iters: int, lr_milestones: list[float], warmup_epochs = 1, n_epochs=100
) -> LRScheduler:

    warmup_iters = warmup_epochs * (total_iters // n_epochs)

    scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-8,
        end_factor=1.0,
        total_iters=warmup_iters
    )

    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        int((total_iters - warmup_iters)*lr_milestones[0]) + 1
    )

    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[scheduler_warmup, scheduler_cosine],
        milestones=[warmup_iters] # El punto (en pasos) donde cambian
    )
    return lr_scheduler


