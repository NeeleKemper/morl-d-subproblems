import torch
from typing import Optional, Generator
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def _flatten_helper(T: int, N: int, tensor: torch.Tensor) -> torch.Tensor:
    """
    Flatten a tensor of shape [T, N, ...] into [T * N, ...].
    """
    return tensor.view(T * N, *tensor.size()[2:])


class RolloutStorage:
    """
    Storage buffer for PPO rollouts, supporting multi‐objective rewards.
    """

    def __init__(
            self,
            num_steps: int,
            num_processes: int,
            obs_shape: tuple[int, ...],
            action_space: torch.distributions.Distribution,
            recurrent_hidden_state_size: int,
            reward_dim: int = 1,
    ) -> None:
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.recurrent_hidden_states = torch.zeros(num_steps + 1, num_processes, recurrent_hidden_state_size)
        self.rewards = torch.zeros(num_steps, num_processes, reward_dim)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, reward_dim)
        self.returns = torch.zeros(num_steps + 1, num_processes, reward_dim)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)

        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.step = 0


    def to(self, device: torch.device) -> None:
        """
        Move all storage tensors to the specified device.
        """
        self.obs = self.obs.to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)

    def insert(
            self,
            obs: torch.Tensor,
            recurrent_hidden_states: torch.Tensor,
            actions: torch.Tensor,
            action_log_probs: torch.Tensor,
            value_preds: torch.Tensor,
            rewards: torch.Tensor,
            masks: torch.Tensor,
            bad_masks: torch.Tensor,
    ) -> None:
        """
        Insert a single timestep of data into storage at index `self.step`.
        """
        self.obs[self.step + 1].copy_(obs)
        self.recurrent_hidden_states[self.step + 1].copy_(recurrent_hidden_states)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        """
        Copy last timestep (T) to index 0 so that next rollout continues smoothly.
        """
        self.obs[0].copy_(self.obs[-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])

    def compute_returns(
            self,
            next_value: torch.Tensor,
            use_gae: bool,
            gamma: float,
            gae_lambda: float,
            use_proper_time_limits: bool = True,
    ) -> None:
        """
        Compute discounted returns (and optionally GAE‐advantages) for multi‐objective rewards.
        """
        if use_proper_time_limits:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1] - \
                            self.value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
                    gae = gae * self.bad_masks[step + 1]
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = (self.returns[step + 1] * gamma * self.masks[step + 1] + self.rewards[step]) * \
                                         self.bad_masks[step + 1] + (1 - self.bad_masks[step + 1]) * self.value_preds[
                                             step]
        else:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1] - \
                            self.value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = self.returns[step + 1] * gamma * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(
            self,
            advantages: Optional[torch.Tensor],
            num_mini_batch: int = None,
            mini_batch_size: int = None
    ) -> Generator[tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[
            torch.Tensor]], None, None]:
        """
        Yield minibatches of data for PPO updates.
        """
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps
        if mini_batch_size is None:
            mini_batch_size = batch_size // num_mini_batch

        sampler = BatchSampler(
            SubsetRandomSampler(range(int(batch_size))),
            int(mini_batch_size),
            drop_last=True)
        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(
                -1, self.recurrent_hidden_states.size(-1))[indices]
            actions_batch = self.actions.view(-1,
                                              self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, *self.value_preds.size()[2:])[indices]
            return_batch = self.returns[:-1].view(-1, *self.returns.size()[2:])[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1,
                                                                    1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ
