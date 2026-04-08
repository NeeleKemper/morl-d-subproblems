import os
import torch
import pickle
import numpy as np
import torch.optim as optim

from copy import deepcopy
from typing import Optional

from agents.single_policy.ppo.ppo import PPO
from agents.single_policy.ppo.a2c_ppo.model import Policy


class Sample:
    def __init__(self,
                 env_params: dict[str, any],
                 actor_critic: Policy,
                 agent: PPO,
                 weights: torch.Tensor,
                 learning_rate: float,
                 eps: float,
                 objs: Optional[np.ndarray] = None):
        self.env_params = env_params
        self.actor_critic = actor_critic
        self.agent = agent
        self.learning_rate = learning_rate
        self.weights = weights
        self.eps = eps
        if self.agent is not None:
            self.link_policy_agent()
        self.objs = objs

    @classmethod
    def copy_from(cls, sample):
        env_params = deepcopy(sample.env_params)
        actor_critic = deepcopy(sample.actor_critic)
        agent = deepcopy(sample.agent)
        learning_rate = sample.learning_rate
        eps = sample.eps
        weights = deepcopy(sample.weights)
        objs = deepcopy(sample.objs)
        return cls(
            env_params=env_params,
            actor_critic=actor_critic,
            agent=agent,
            learning_rate=learning_rate,
            eps=eps,
            weights=weights,
            objs=objs
        )

    def set_weights(self, weights: torch.Tensor):
        self.weights = weights

    def link_policy_agent(self):
        self.agent.actor_critic = self.actor_critic
        optim_state_dict = deepcopy(self.agent.optimizer.state_dict())
        self.agent.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.learning_rate, eps=self.eps)
        self.agent.optimizer.load_state_dict(optim_state_dict)

    def save(self, path: str, file_name: str) -> None:
        torch.save(
            self.actor_critic.state_dict(),
            os.path.join(path, f'{file_name}_actor_critic.pt')
        )

        torch.save(
            self.agent.optimizer.state_dict(),
            os.path.join(path, f'{file_name}_optimizer.pt')
        )
        with open(os.path.join(path, f'{file_name}_env_params.pkl'), 'wb') as f:
            pickle.dump(self.env_params, f)

        np.savetxt(
            os.path.join(path, f'{file_name}_objs.txt'),
            self.objs,
            delimiter=',',
            fmt='%.6f'
        )
