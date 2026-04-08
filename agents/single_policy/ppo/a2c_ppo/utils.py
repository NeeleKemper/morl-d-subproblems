import torch
import torch.nn as nn

from torch.optim import Optimizer
from typing import Callable, Optional

from agents.single_policy.ppo.a2c_ppo.envs import VecNormalize


def get_render_func(venv: any) -> Optional[Callable]:
    """
    Recursively search through VecEnv wrappers to find the base environment's `render` method.
    """
    if hasattr(venv, 'envs'):
        return venv.envs[0].render
    elif hasattr(venv, 'venv'):
        return get_render_func(venv.venv)
    elif hasattr(venv, 'env'):
        return get_render_func(venv.env)

    return None


def get_vec_normalize(venv: any) -> Optional[VecNormalize]:
    """
    Recursively search through VecEnv wrappers to find an instance of VecNormalize.
    """
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, 'venv'):
        return get_vec_normalize(venv.venv)

    return None


# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    """
    A simple layer that adds a learnable bias to its input.

    This is used in KFAC implementations to handle biases separately from weights.

    The bias parameter is stored as shape [out_features, 1], but in forward
    it is broadcast either to a 2D or 4D shape, depending on `x.dim()`.
    """

    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add bias to `x`. Supports both 2D inputs ([N, F]) and 4D inputs ([N, C, H, W]).
        """
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def update_linear_schedule(
        optimizer: Optimizer,
        epoch: float,
        total_num_epochs: int,
        initial_lr: float
) -> None:
    """
    Linearly decrease the optimizer's learning rate from `initial_lr` to 0 over `total_steps`.
    """
    lr = max(initial_lr - (initial_lr * (epoch / float(total_num_epochs))), initial_lr * 0.1)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init(
        module: nn.Module,
        weight_init: Callable[[torch.Tensor, float], None],
        bias_init: Callable[[torch.Tensor], None],
        gain: float = 1.0
) -> nn.Module:
    """
    Apply orthogonal or Xavier initialization to a module's weights and constant init to its biases.
    """
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module
