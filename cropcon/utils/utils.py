import os as os
import random
from pathlib import Path
from torch.optim.optimizer import Optimizer

import numpy as np
import torch


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
