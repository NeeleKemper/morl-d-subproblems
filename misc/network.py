import torch
import torch.nn as nn
from typing import Iterable


@torch.no_grad()
def polyak_update(params: Iterable[nn.Parameter], target_params: Iterable[nn.Parameter], tau: float) -> None:
    for param, target_param in zip(params, target_params):
        if tau == 1:
            target_param.data.copy_(param.data)
        else:
            target_param.data.mul_(1.0 - tau)
            torch.add(target_param.data, param.data, alpha=tau, out=target_param.data)


def mlp(input_dim: int, output_dim: int, net_arch: list[int], activation_fn: type[nn.Module] = nn.ReLU,
        drop_rate: float = 0.0, layer_norm: bool = False, ) -> nn.Sequential:
    assert len(net_arch) > 0
    modules = [nn.Linear(input_dim, net_arch[0])]
    if drop_rate > 0.0:
        modules.append(nn.Dropout(p=drop_rate))
    if layer_norm:
        modules.append(nn.LayerNorm(net_arch[0]))
    modules.append(activation_fn())

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        if drop_rate > 0.0:
            modules.append(nn.Dropout(p=drop_rate))
        if layer_norm:
            modules.append(nn.LayerNorm(net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1]
        modules.append(nn.Linear(last_layer_dim, output_dim))

    return nn.Sequential(*modules)


@torch.no_grad()
def layer_init(layer, method='orthogonal', weight_gain: float = 1, bias_const: float = 0) -> None:
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        if method == 'xavier':
            nn.init.xavier_uniform_(layer.weight, gain=weight_gain)
        elif method == 'orthogonal':
            nn.init.orthogonal_(layer.weight, gain=weight_gain)
        nn.init.constant_(layer.bias, bias_const)


@torch.no_grad()
def polyak_update(
        params: Iterable[torch.nn.Parameter],
        target_params: Iterable[torch.nn.Parameter],
        tau: float,
) -> None:
    for param, target_param in zip(params, target_params):
        if tau == 1:
            target_param.data.copy_(param.data)
        else:
            target_param.data.mul_(1.0 - tau)
            torch.add(target_param.data, param.data, alpha=tau, out=target_param.data)
