import torch
import torch.nn as nn
from torch.distributions import Normal
from misc.network import layer_init, mlp


class Actor(nn.Module):
    def __init__(
            self,
            num_obs: int,
            num_outputs: int,
            net_arch: list[int],
            layernorm: bool
    ):
        super().__init__()
        # Build an MLP with optional LayerNorm between each Linear and activation
        self.net = mlp(
            input_dim=num_obs,
            output_dim=num_outputs,
            net_arch=net_arch,
            activation_fn=nn.Tanh,
            drop_rate=0.0,
            layer_norm=layernorm,
        )
        # Log‐standard‐deviation parameter (learned)
        self.logstd = nn.Parameter(torch.zeros(1, num_outputs))

        # Initialize all hidden Linear layers with orthogonal gain=1.0 (instead of sqrt(2))
        linears = [m for m in self.net if isinstance(m, nn.Linear)]
        for layer in linears[:-1]:
            layer_init(layer, method='orthogonal', weight_gain=1.0, bias_const=0.0)
        # Final actor‐output layer with small gain=0.01
        layer_init(linears[-1], method='orthogonal', weight_gain=0.01, bias_const=0.0)

    def forward(self, obs: torch.Tensor):
        action_mean = self.net(obs)
        action_logstd = self.logstd.expand_as(action_mean)
        return action_mean, action_logstd


class Critic(nn.Module):
    def __init__(
            self,
            num_obs: int,
            num_outputs: int,
            net_arch: list[int],
            layernorm: bool
    ):
        super().__init__()
        # Build an MLP with optional LayerNorm between each Linear and activation
        self.net = mlp(
            input_dim=num_obs,
            output_dim=num_outputs,
            net_arch=net_arch,
            activation_fn=nn.Tanh,
            drop_rate=0.0,
            layer_norm=layernorm,
        )
        # Initialize every Linear in the critic with orthogonal gain=1.0
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                layer_init(layer, method='orthogonal', weight_gain=1.0, bias_const=0.0)

    def forward(self, obs: torch.Tensor):
        return self.net(obs)


class MOMLPBase(nn.Module):
    def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            net_arch: list[int],
            layernorm: bool,
            reward_dim: int
    ):
        super().__init__()
        self.actor = Actor(obs_dim, action_dim, net_arch, layernorm)
        self.critic = Critic(obs_dim, reward_dim, net_arch, layernorm)

    def forward(self, obs: torch.Tensor):
        action_mean, action_logstd = self.actor(obs)
        critic_value = self.critic(obs)
        return critic_value, action_mean, action_logstd


class Policy(nn.Module):
    def __init__(
            self,
            obs_shape,
            action_space,
            net_arch: list[int] = [64, 64],
            layernorm: bool = True,
            reward_dim: int = 1,
    ):
        super().__init__()
        action_dim = action_space.shape[0]
        self.base = MOMLPBase(
            obs_dim=obs_shape[0],
            action_dim=action_dim,
            net_arch=net_arch,
            layernorm=layernorm,
            reward_dim=reward_dim
        )

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        value, mu, logstd = self.base(obs)
        std = torch.exp(logstd)
        dist = Normal(mu, std)

        action = dist.mean if deterministic else dist.sample()
        logp = dist.log_prob(action).sum(-1, keepdim=True)
        return value, action, logp

    def get_value(self, obs: torch.Tensor):
        value, _, _ = self.base(obs)
        return value

    def evaluate_actions(
            self,
            obs: torch.Tensor,
            action: torch.Tensor,
    ):
        value, mu, logstd = self.base(obs)
        std = torch.exp(logstd)
        dist = Normal(mu, std)
        logp = dist.log_prob(action).sum(-1, keepdim=True)
        entropy = dist.entropy().mean()
        return value, logp, entropy