import torch
import torch.nn as nn
from misc.network import mlp, layer_init

LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, action_space: any, net_arch: list = [256, 256]):
        super().__init__()
        self.action_space = action_space
        self.latent_pi = mlp(obs_dim, -1, net_arch)
        self.mean = nn.Linear(net_arch[-1], action_dim)
        self.log_std_linear = nn.Linear(net_arch[-1], action_dim)

        self.register_buffer('action_scale',
                             torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32))
        self.register_buffer('action_bias',
                             torch.tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32))

        self.apply(layer_init)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.latent_pi(obs)
        mean = self.mean(h)
        log_std = self.log_std_linear(h)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(obs)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return mean

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=1)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class SoftQNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, output_dim: int, net_arch: list = [256, 256]):
        super().__init__()
        self.net = mlp(obs_dim + action_dim, output_dim, net_arch, activation_fn=nn.ReLU)
        self.apply(layer_init)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q_values = self.net(torch.cat((obs, action), dim=1))
        return q_values



