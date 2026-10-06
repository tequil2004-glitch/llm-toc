"""Rollout buffer for PPO-based MARL.

Stores per-agent transitions for a fixed horizon of ``num_steps`` across
``num_envs`` parallel environments. The buffer supports both IPPO
(decentralized critic) and MAPPO (centralized critic) by storing the
centralized state alongside the decentralized actor observation.

Storage layout: every tensor is shaped ``(num_steps, num_envs, ...)``
so that minibatches can be drawn by flattening the first two dims and
shuffling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import torch


@dataclass
class RolloutBatch:
    """A minibatch sampled from the :class:`RolloutBuffer`.

    All tensors are already flattened to ``(B, ...)`` and moved to the
    requested device.
    """

    actor_obs: torch.Tensor
    critic_obs: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    old_values: torch.Tensor


class RolloutBuffer:
    """Fixed-horizon rollout buffer for multi-agent PPO.

    The buffer is *per-agent*: each agent in the team has its own
    :class:`RolloutBuffer`. The trainer creates ``num_agents`` buffers.
    """

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        actor_obs_dim: int,
        critic_obs_dim: int,
        action_dim: int,
        device: torch.device,
    ) -> None:
        """Initialize the buffer.

        Args:
            num_steps: Rollout horizon (steps per env per rollout).
            num_envs: Number of parallel environments.
            actor_obs_dim: Decentralized actor observation dimension.
            critic_obs_dim: Critic (possibly centralized) observation dim.
            action_dim: Continuous action dimensionality.
            device: Torch device for sampled tensors.
        """
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.actor_obs_dim = actor_obs_dim
        self.critic_obs_dim = critic_obs_dim
        self.action_dim = action_dim
        self.device = device

        # Pre-allocate storage on CPU; samples are moved to device on get.
        self.actor_obs = np.zeros(
            (num_steps, num_envs, actor_obs_dim), dtype=np.float32)
        self.critic_obs = np.zeros(
            (num_steps, num_envs, critic_obs_dim), dtype=np.float32)
        self.actions = np.zeros(
            (num_steps, num_envs, action_dim), dtype=np.float32)
        self.log_probs = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.rewards = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.values = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.dones = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.advantages = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.returns = np.zeros((num_steps, num_envs), dtype=np.float32)

        self.ptr = 0

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Zero all storage and reset the write pointer."""
        self.actor_obs.fill(0)
        self.critic_obs.fill(0)
        self.actions.fill(0)
        self.log_probs.fill(0)
        self.rewards.fill(0)
        self.values.fill(0)
        self.dones.fill(0)
        self.advantages.fill(0)
        self.returns.fill(0)
        self.ptr = 0

    # ------------------------------------------------------------------
    def insert(
        self,
        actor_obs: np.ndarray,
        critic_obs: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        """Insert a single transition for all envs.

        Args:
            actor_obs: ``(num_envs, actor_obs_dim)`` decentralized obs.
            critic_obs: ``(num_envs, critic_obs_dim)`` critic obs.
            actions: ``(num_envs, action_dim)`` actions taken.
            log_probs: ``(num_envs,)`` log-probabilities of actions.
            rewards: ``(num_envs,)`` team rewards (collective return).
            values: ``(num_envs,)`` value estimates V(s_t).
            dones: ``(num_envs,)`` episode-termination flags.

        Raises:
            IndexError: If the buffer is already full.
        """
        if self.ptr >= self.num_steps:
            raise IndexError("RolloutBuffer is full; call reset() first.")
        t = self.ptr
        self.actor_obs[t] = actor_obs
        self.critic_obs[t] = critic_obs
        self.actions[t] = actions
        self.log_probs[t] = log_probs
        self.rewards[t] = rewards
        self.values[t] = values
        self.dones[t] = dones
        self.ptr += 1

    # ------------------------------------------------------------------
    def compute_advantages(
        self,
        last_values: np.ndarray,
        gamma: float,
        gae_lambda: float,
        normalize: bool = True,
    ) -> None:
        """Compute GAE advantages and discounted returns in-place.

        Args:
            last_values: ``(num_envs,)`` bootstrap value V(s_{T+1}).
            gamma: Discount factor.
            gae_lambda: GAE lambda.
            normalize: If True, standardize advantages (zero mean, unit
                variance) for stable PPO updates.
        """
        last_gae = np.zeros(self.num_envs, dtype=np.float32)
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_values = last_values.astype(np.float32)
            else:
                next_values = self.values[t + 1]
            next_non_terminal = 1.0 - self.dones[t]
            delta = (
                self.rewards[t]
                + gamma * next_values * next_non_terminal
                - self.values[t]
            )
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae
        self.returns = self.advantages + self.values

        if normalize:
            adv = self.advantages
            mean = adv.mean()
            std = adv.std()
            self.advantages = (adv - mean) / (std + 1e-8)

    # ------------------------------------------------------------------
    def get_minibatches(
        self,
        num_minibatches: int,
        rng: np.random.Generator,
    ) -> Sequence[RolloutBatch]:
        """Yield minibatches as :class:`RolloutBatch` objects.

        Args:
            num_minibatches: Number of minibatches to split the data into.
            rng: NumPy RNG for shuffling indices.

        Returns:
            A list of :class:`RolloutBatch` objects covering all
            transitions exactly once.
        """
        if self.ptr != self.num_steps:
            raise RuntimeError(
                f"Buffer has {self.ptr}/{self.num_steps} transitions; "
                f"must be full before sampling."
            )
        batch = self.num_steps * self.num_envs
        indices = np.arange(batch)
        rng.shuffle(indices)
        # Truncate to a multiple of num_minibatches to avoid ragged splits.
        mb_size = batch // num_minibatches
        indices = indices[: mb_size * num_minibatches].reshape(
            num_minibatches, mb_size)

        def _take(arr: np.ndarray, idx: np.ndarray) -> torch.Tensor:
            flat = arr.reshape(batch, *arr.shape[2:])
            return torch.from_numpy(flat[idx]).to(self.device).float()

        batches = []
        for mb_idx in indices:
            batches.append(
                RolloutBatch(
                    actor_obs=_take(self.actor_obs, mb_idx),
                    critic_obs=_take(self.critic_obs, mb_idx),
                    actions=_take(self.actions, mb_idx),
                    old_log_probs=_take(self.log_probs, mb_idx),
                    advantages=_take(self.advantages, mb_idx),
                    returns=_take(self.returns, mb_idx),
                    old_values=_take(self.values, mb_idx),
                )
            )
        return batches

    # ------------------------------------------------------------------
    def statistics(self) -> Dict[str, float]:
        """Return summary statistics for logging."""
        return {
            "reward_mean": float(self.rewards.mean()),
            "reward_std": float(self.rewards.std()),
            "value_mean": float(self.values.mean()),
            "advantage_mean": float(self.advantages.mean()),
            "advantage_std": float(self.advantages.std()),
        }
