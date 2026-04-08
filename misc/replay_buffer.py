import torch
import numpy as np


def _to_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.array(x)


class ReplayBuffer:
    def __init__(
            self,
            obs_shape,
            action_dim,
            rew_dim=1,
            max_size=100000,
            obs_dtype=np.float32,
            action_dtype=np.float32,
            seed: int = 42
    ):
        self.max_size = max_size
        self.obs_shape = obs_shape
        self.obs_dtype = obs_dtype
        self.action_dim = action_dim
        self.action_dtype = action_dtype
        self.rew_dim = rew_dim
        self.ptr, self.size = 0, 0
        self.obs = np.zeros((max_size,) + obs_shape, dtype=obs_dtype)
        self.next_obs = np.zeros((max_size,) + obs_shape, dtype=obs_dtype)
        self.actions = np.zeros((max_size, action_dim), dtype=action_dtype)
        self.rewards = np.zeros((max_size, rew_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)

        self.rng = np.random.default_rng(seed)

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = _to_np(obs)
        self.actions[self.ptr] = _to_np(action)
        self.next_obs[self.ptr] = _to_np(next_obs)
        self.rewards[self.ptr] = _to_np(reward)
        self.dones[self.ptr] = _to_np(done)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, replace=True, use_cer=False, to_tensor=False, device=None) -> tuple:
        inds = self.rng.choice(self.size, batch_size, replace=replace)
        if use_cer:
            inds[0] = self.ptr - 1
        experience_tuples = (
            self.obs[inds],
            self.actions[inds],
            self.rewards[inds],
            self.next_obs[inds],
            self.dones[inds],
        )
        if to_tensor:
            return tuple(map(lambda x: torch.tensor(x, device=device), experience_tuples))
        else:
            return experience_tuples

    def sample_obs(self, batch_size, replace=True, to_tensor=False, device=None):
        inds = self.rng.choice(self.size, batch_size, replace=replace)
        if to_tensor:
            return torch.tensor(self.obs[inds], device=device)
        else:
            return self.obs[inds]

    def get_all_data(self, max_samples=None):
        if max_samples is not None:
            inds = self.rng.choice(self.size, min(max_samples, self.size), replace=False)
        else:
            inds = np.arange(self.size)
        return (
            self.obs[inds],
            self.actions[inds],
            self.rewards[inds],
            self.next_obs[inds],
            self.dones[inds],
        )

    def __len__(self):
        return self.size

    def reset(self):
        self.ptr, self.size = 0, 0
        self.obs = np.zeros((self.max_size,) + self.obs_shape, dtype=self.obs_dtype)
        self.next_obs = np.zeros((self.max_size,) + self.obs_shape, dtype=self.obs_dtype)
        self.actions = np.zeros((self.max_size, self.action_dim), dtype=self.action_dtype)
        self.rewards = np.zeros((self.max_size, self.rew_dim), dtype=np.float32)
        self.dones = np.zeros((self.max_size, 1), dtype=np.float32)
