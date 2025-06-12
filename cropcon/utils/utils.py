import os as os
import random
from pathlib import Path
from torch.optim.optimizer import Optimizer

import numpy as np
import torch
import torchvision.transforms.v2.functional as Fv


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_generator(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# to make flops calculator work
def prepare_input(input_res):
    image = {}
    x1 = torch.FloatTensor(*input_res)
    # input_res[-2] = 2
    input_res = list(input_res)
    input_res[-3] = 2
    x2 = torch.FloatTensor(*tuple(input_res))
    image["optical"] = x1
    image["sar"] = x2
    return dict(img=image)


def get_best_model_ckpt_path(exp_dir: str | Path) -> str:
    return os.path.join(
        exp_dir, next(f for f in os.listdir(exp_dir) if f.endswith("_best.pth"))
    )

def get_final_model_ckpt_path(exp_dir: str | Path) -> str:
    return os.path.join(
        exp_dir, next(f for f in os.listdir(exp_dir) if f.endswith("_final.pth"))
    )

class LARS(torch.optim.Optimizer):
    def __init__(self, optimizer, eps=1e-8, trust_coef=0.001):
        self.optimizer = optimizer
        self.eps = eps
        self.trust_coef = trust_coef

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @property
    def state(self):
        return self.optimizer.state

    @property
    def defaults(self):
        return self.optimizer.defaults

    @property
    def _state_dict_hooks(self):
        return self.optimizer._state_dict_hooks

    @property
    def _load_state_dict_pre_hooks(self):
        return self.optimizer._load_state_dict_pre_hooks

    @property
    def _load_state_dict_post_hooks(self):
        return self.optimizer._load_state_dict_post_hooks

    @param_groups.setter
    def param_groups(self, value):
        self.optimizer.param_groups = value

    def step(self, closure=None):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                param_norm = p.data.norm()
                grad_norm = p.grad.data.norm()
                if param_norm > 0 and grad_norm > 0:
                    local_lr = self.trust_coef * param_norm / (grad_norm + self.eps)
                    p.grad.data.mul_(local_lr)
        self.optimizer.step(closure)

    def zero_grad(self, set_to_none=False):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def register_state_dict_pre_hook(self, hook):
        return self.optimizer.register_state_dict_pre_hook(hook)

    def register_state_dict_post_hook(self, hook):
        return self.optimizer.register_state_dict_post_hook(hook)


class RandomChannelDropout(torch.nn.Module):
    def __init__(self, p=0.5, max_drop=1):
        super().__init__()
        self.p = p
        self.max_drop = max_drop

    def forward(self, x, drop_indices=None):
        """
        x: [T, C, H, W] or [C, H, W]
        drop_indices: if provided, drops the same indices
        """
        if drop_indices is None and torch.rand(1).item() < self.p:
            C = x.shape[1]
            num_drop = torch.randint(1, self.max_drop + 1, ())
            drop_indices = torch.randperm(C)[:num_drop]

        if drop_indices is not None:
            # Zero out selected channels out-of-place
            mask = torch.ones_like(x)
            mask[:, drop_indices, :, :] = 0
            x = x * mask

        return x, drop_indices


class ConsistentTransform(torch.nn.Module):
    def __init__(self, degrees=30, p=0.5):
        super().__init__()
        self.degrees = degrees
        self.hflip_p = p
        self.vflip_p = p

    def forward(self, sample):
        img, mask = sample["image"], sample["mask"]
        # Sample random parameters
        angle = torch.empty(1).uniform_(-self.degrees, self.degrees).item()
        do_hflip = torch.rand(1).item() < self.hflip_p
        do_vflip = torch.rand(1).item() < self.vflip_p

        # Apply same transform to all frames
        img = Fv.rotate(img, angle)
        mask = Fv.rotate(mask, angle)

        if do_hflip:
            img = Fv.hflip(img)
            mask = Fv.hflip(mask)
        if do_vflip:
            img = Fv.vflip(img)
            mask = Fv.vflip(mask)

        return {"image": img, "mask": mask}