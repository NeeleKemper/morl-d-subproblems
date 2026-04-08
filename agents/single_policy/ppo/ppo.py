import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from typing import Optional

from agents.single_policy.ppo.a2c_ppo.model import Policy
from agents.single_policy.ppo.a2c_ppo.storage import RolloutStorage


class PPO:
    def __init__(self,
                 actor_critic: Policy,
                 clip_param: float,
                 ppo_epoch: int,
                 num_mini_batches: int,
                 value_loss_coef: float,
                 entropy_coef: float,
                 lr: float,
                 eps: float,
                 max_grad_norm: float,
                 use_clipped_value_loss: bool,
                 ):
        self.actor_critic = actor_critic
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

    def _compute_advantages(self, returns, value_preds, weights):
        g_ret = (weights * returns[:-1]).sum(dim=-1)
        g_val = (weights * value_preds[:-1]).sum(dim=-1)
        return g_ret - g_val

    def update(
            self,
            rollouts: RolloutStorage,
            weights: torch.Tensor,
            obj_var: Optional[np.ndarray] = None
    ) -> tuple[float, float, float]:
        op_axis = tuple(range(len(rollouts.returns.shape) - 1))

        returns = rollouts.returns * torch.Tensor(
            np.sqrt(obj_var + 1e-8)) if obj_var is not None else rollouts.returns
        value_preds = rollouts.value_preds * torch.Tensor(
            np.sqrt(obj_var + 1e-8)) if obj_var is not None else rollouts.value_preds

        advantages = self._compute_advantages(returns, value_preds, weights)

        advantages = (advantages - advantages.mean(axis=op_axis)) / (advantages.std(axis=op_axis) + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):

            data_generator = rollouts.feed_forward_generator(advantages, self.num_mini_batches)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                    value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                    adv_targ = sample

                values, action_log_probs, dist_entropy = self.actor_critic.evaluate_actions(obs_batch, actions_batch)

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                                         (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()
                loss = value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batches

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
