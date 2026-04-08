import torch
import numpy as np
import gymnasium as gym
from abc import ABC, abstractmethod
from misc.utils import seed_everything


class Agent(ABC):
    def __init__(self, env: gym.Env, seed: int, device: str, name: str):
        seed_everything(seed)
        self.seed = seed
        self.env = env
        self.env.action_space.seed(seed)
        self.env.observation_space.seed(seed)

        if isinstance(env.unwrapped.observation_space, gym.spaces.Discrete):
            self.obs_shape = (1,)
            self.obs_dim = 1
        else:
            self.obs_shape = env.unwrapped.observation_space.shape
            self.obs_dim = env.unwrapped.observation_space.shape[0]

        if isinstance(env.unwrapped.action_space, gym.spaces.Discrete):
            self.action_shape = (1,)  # storage shape
            self.action_dim = env.unwrapped.action_space.n  # network output dim
        elif isinstance(env.unwrapped.action_space, gym.spaces.MultiBinary):
            self.action_shape = (env.unwrapped.action_space.n,)
            self.action_dim = env.unwrapped.action_space.n
        else:
            self.action_shape = env.unwrapped.action_space.shape
            self.action_dim = env.unwrapped.action_space.shape[0]

        self.action_space = env.unwrapped.action_space
        self.reward_dim = env.unwrapped.reward_space.shape[0]

        self.device = torch.device(
            device if device in ['cpu', 'cuda'] else ('cuda' if torch.cuda.is_available() else 'cpu'))

        self.name = name
        self.global_step = 0
        self.num_episodes = 0
        self.np_random = np.random.default_rng(self.seed)

    @abstractmethod
    def save(self, path: str, file_name: str, save_replay_buffer: bool = True) -> None:
        pass

    @abstractmethod
    def load(self, path: str, file_name: str) -> None:
        pass

    @abstractmethod
    def save_config(self, path: str, file_name: str) -> None:
        pass
