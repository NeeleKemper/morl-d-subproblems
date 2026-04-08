import os
import torch

import numpy as np
import gymnasium as gym
import mo_gymnasium as mo_gym

from typing import Optional, Callable

from agents.single_policy.ppo.externals.bench.monitor import Monitor
from agents.single_policy.ppo.externals.common.vec_env.vec_env import VecEnvWrapper
from agents.single_policy.ppo.externals.common.vec_env.dummy_vec_env import DummyVecEnv
from agents.single_policy.ppo.externals.common.vec_env.shmem_vec_env import ShmemVecEnv
from agents.single_policy.ppo.externals.common.vec_env.vec_normalize import VecNormalize as VecNormalize_

try:
    import dm_control2gym
except ImportError:
    pass

try:
    import roboschool
except ImportError:
    pass

try:
    import pybullet_envs
except ImportError:
    pass


def make_env(
        env_id: str,
        seed: int,
        rank: int,
        log_dir: Optional[str],
        allow_early_resets: bool,
        max_episode_steps: int = 500
) -> Callable[[], gym.Env]:
    """
    Create a single‐process environment factory that:
    """

    def _thunk():
        env = mo_gym.make(env_id, max_episode_steps=max_episode_steps)
        env.reset(seed=seed + rank)
        env.action_space.seed(seed + rank)
        env.observation_space.seed(seed + rank)
        env.seed = seed + rank
        env = TimeLimitMask(env, reset_seed=seed + rank)

        if log_dir is not None:
            env = Monitor(
                env,
                os.path.join(log_dir, str(rank)),
                allow_early_resets=allow_early_resets)
        else:
            env = Monitor(
                env,
                None,
                allow_early_resets=allow_early_resets)
        return env

    return _thunk


def make_vec_envs(
        env_name: str,
        seed: int,
        num_processes: int,
        gamma: Optional[float],
        log_dir: Optional[str],
        device: torch.device,
        allow_early_resets: bool,
        num_frame_stack: Optional[int] = None,
        obj_rms: bool = False,
        ob_rms: bool = False,
        max_episode_steps: int = 500,
        multiprocessing_envs: bool = True,
) -> VecEnvWrapper:
    """
    Create a vectorized set of environments (either ShmemVecEnv or DummyVecEnv),
    optionally normalize observations/returns, and wrap to produce PyTorch tensors on `device`.
    """
    envs = [
        make_env(env_name, seed, i, log_dir, allow_early_resets, max_episode_steps)
        for i in range(num_processes)
    ]

    if len(envs) > 1 and multiprocessing_envs:
        envs = ShmemVecEnv(envs, context='spawn')
    else:
        envs = DummyVecEnv(envs)

    if len(envs.observation_space.shape) == 1:
        if gamma is None:
            envs = VecNormalize(envs, ret=False, obj_rms=obj_rms, ob=ob_rms)
        else:
            envs = VecNormalize(envs, gamma=gamma, obj_rms=obj_rms, ob=ob_rms)

    envs = VecPyTorch(envs, device)

    if num_frame_stack is not None:
        envs = VecPyTorchFrameStack(envs, num_frame_stack, device)
    elif len(envs.observation_space.shape) == 3:
        envs = VecPyTorchFrameStack(envs, 4, device)

    return envs


# Checks whether done was caused my timit limits or not
class TimeLimitMask(gym.Wrapper):
    def __init__(self, env=None, reset_seed=None):
        super().__init__(env)
        self.seed = reset_seed

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info['bad_transition'] = True
        info['obj'] = rew
        rew = 0.
        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(seed=self.seed, **kwargs)[0]


# Can be used to test recurrent policies for Reacher-v2
class MaskGoal(gym.ObservationWrapper):
    def observation(self, observation):
        if self.env._elapsed_steps > 0:
            observation[-2:] = 0
        return observation


class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super(TransposeObs, self).__init__(env)


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


class VecNormalize(VecNormalize_):
    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs, update=True):
        if self.ob_rms:
            if self.training and update:
                self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) /
                          np.sqrt(self.ob_rms.var + self.epsilon),
                          -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


# Derived from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_frame_stack.py
class VecPyTorchFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack, device=None):
        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space  # wrapped ob space
        self.shape_dim0 = wos.shape[0]

        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)

        if device is None:
            device = torch.device('cpu')
        self.stacked_obs = torch.zeros((venv.num_envs,) +
                                       low.shape).to(device)

        observation_space = gym.spaces.Box(
            low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stacked_obs[:, :-self.shape_dim0] = \
            self.stacked_obs[:, self.shape_dim0:]
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        if torch.backends.cudnn.deterministic:
            self.stacked_obs = torch.zeros(self.stacked_obs.shape)
        else:
            self.stacked_obs.zero_()
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs

    def close(self):
        self.venv.close()
