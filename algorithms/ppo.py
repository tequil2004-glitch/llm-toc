"""Core PPO update logic shared by IPPO and MAPPO.

The :class:`PPOUpdater` performs the canonical clipped-surrogate PPO
update with the strictly-enforced hyperparameters from
:class:`~llm_toc.config.PPOConfig`:

* Learning rate ``3e-4`` with linear decay.
* PPO clip ``0.2``.
* Entropy coefficient ``0.01``.
* Discount ``0.99``, GAE lambda ``0.95``.
* 10 PPO epochs, max grad norm ``0.5``.

The updater is observation-agnostic: it operates on an
:class:`~llm_toc.models.actor_critic.ActorCritic` module and a list of
:class:`~llm_toc.algorithms.buffer.RolloutBatch` objects. Whether the
critic is decentralized (IPPO) or centralized (MAPPO) is determined by
the contents of the rollout buffer, not by this updater.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from llm_toc.algorithms.buffer import RolloutBatch
from llm_toc.config import PPOConfig
from llm_toc.models.actor_critic import ActorCritic


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------
def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    last_values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Generalized Advantage Estimation.

    Args:
        rewards: ``(T, N)`` rewards.
        values: ``(T, N)`` value estimates V(s_t).
        last_values: ``(N,)`` bootstrap value V(s_{T+1}).
        dones: ``(T, N)`` terminal flags.
        gamma: Discount factor.
        gae_lambda: GAE lambda.

    Returns:
        Tuple ``(advantages, returns)`` each ``(T, N)``.
    """
    T, N = rewards.shape
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = np.zeros(N, dtype=np.float32)
    for t in reversed(range(T)):
        if t == T - 1:
            next_values = last_values.astype(np.float32)
        else:
            next_values = values[t + 1]
        next_non_terminal = 1.0 - dones[t]
        delta = (
            rewards[t] + gamma * next_values * next_non_terminal - values[t]
        )
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


# ---------------------------------------------------------------------------
# Update statistics
# ---------------------------------------------------------------------------
@dataclass
class PPOUpdateStats:
    """Aggregate PPO update statistics for logging."""

    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_frac: float
    explained_variance: float
    grad_norm: float
    lr: float

    def as_dict(self) -> Dict[str, float]:
        """Return a flat dict suitable for CSV/TensorBoard logging."""
        return {
            "policy_loss": self.policy_loss,
            "value_loss": self.value_loss,
            "entropy": self.entropy,
            "approx_kl": self.approx_kl,
            "clip_frac": self.clip_frac,
            "explained_variance": self.explained_variance,
            "grad_norm": self.grad_norm,
            "lr": self.lr,
        }


# ---------------------------------------------------------------------------
# Updater
# ---------------------------------------------------------------------------
class PPOUpdater:
    """Performs clipped-surrogate PPO updates on an actor-critic module.

    The updater is stateful: it owns the optimizer and tracks the linear
    learning-rate decay schedule across ``total_updates`` calls to
    :meth:`update`. It also enforces the configured ``max_grad_norm``.
    """

    def __init__(
        self,
        module: ActorCritic,
        cfg: PPOConfig,
        total_updates: int,
        device: torch.device,
    ) -> None:
        """Initialize the updater.

        Args:
            module: The actor-critic module to optimize.
            cfg: Strict PPO hyperparameters.
            total_updates: Total number of :meth:`update` calls expected
                over training. Used for linear LR decay.
            device: Torch device.
        """
        self.module = module
        self.cfg = cfg
        self.device = device
        self.total_updates = max(int(total_updates), 1)
        self.current_update = 0
        self.optimizer = Adam(module.parameters(), lr=cfg.lr, eps=1e-5)

    # ------------------------------------------------------------------
    def _current_lr(self) -> float:
        """Compute the linearly-decayed learning rate for this update."""
        if not self.cfg.anneal_lr:
            return self.cfg.lr
        frac = 1.0 - self.current_update / self.total_updates
        return self.cfg.lr * frac

    def _set_lr(self, lr: float) -> None:
        """Set the optimizer's learning rate for all param groups."""
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    # ------------------------------------------------------------------
    def update(
        self,
        batches: Sequence[RolloutBatch],
        rng: np.random.Generator,
    ) -> PPOUpdateStats:
        """Run ``ppo_epochs`` epochs of PPO over the given minibatches.

        Args:
            batches: Pre-shuffled minibatches covering one rollout.
            rng: NumPy RNG (unused but reserved for future stochasticity).

        Returns:
            :class:`PPOUpdateStats` averaged across all epochs/minibatches.
        """
        cfg = self.cfg
        lr = self._current_lr()
        self._set_lr(lr)

        module = self.module
        module.train()

        # Accumulators (running means).
        n_steps = 0
        sum_policy_loss = 0.0
        sum_value_loss = 0.0
        sum_entropy = 0.0
        sum_approx_kl = 0.0
        sum_clip_frac = 0.0
        sum_grad_norm = 0.0

        # For explained variance across the full rollout.
        all_returns: list[torch.Tensor] = []
        all_values: list[torch.Tensor] = []

        for _ in range(cfg.ppo_epochs):
            for batch in batches:
                actor_obs = batch.actor_obs
                critic_obs = batch.critic_obs
                actions = batch.actions
                old_log_probs = batch.old_log_probs
                advantages = batch.advantages
                returns = batch.returns
                old_values = batch.old_values

                new_values = module.critic(critic_obs)
                new_log_probs, entropy = module.actor.evaluate_actions(
                    actor_obs, actions)

                # Re-normalize advantages inside each minibatch for
                # additional stability (optional but standard).
                if cfg.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8)

                log_ratio = new_log_probs - old_log_probs
                ratio = torch.exp(log_ratio)

                # Policy loss (clipped surrogate).
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps,
                                    1.0 + cfg.clip_eps) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped value loss is optional; use MSE here).
                value_pred_clipped = old_values + torch.clamp(
                    new_values - old_values, -cfg.clip_eps, cfg.clip_eps)
                value_loss_unclipped = (new_values - returns).pow(2)
                value_loss_clipped = (value_pred_clipped - returns).pow(2)
                value_loss = 0.5 * torch.max(
                    value_loss_unclipped, value_loss_clipped).mean()

                entropy_loss = -entropy.mean()
                total_loss = (
                    policy_loss
                    + cfg.value_coef * value_loss
                    + cfg.ent_coef * entropy_loss
                )

                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    module.parameters(), cfg.max_grad_norm).item()
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                    clip_frac = (
                        ((ratio - 1.0).abs() > cfg.clip_eps)
                        .float().mean().item()
                    )

                n_steps += 1
                sum_policy_loss += float(policy_loss.item())
                sum_value_loss += float(value_loss.item())
                sum_entropy += float(-entropy_loss.item())
                sum_approx_kl += float(approx_kl)
                sum_clip_frac += float(clip_frac)
                sum_grad_norm += float(grad_norm)

                all_returns.append(returns.detach())
                all_values.append(new_values.detach())

        self.current_update += 1
        module.eval()

        # Explained variance over the full rollout sample.
        ret_cat = torch.cat(all_returns, dim=0)
        val_cat = torch.cat(all_values, dim=0)
        ev = _explained_variance(ret_cat, val_cat)

        return PPOUpdateStats(
            policy_loss=sum_policy_loss / max(n_steps, 1),
            value_loss=sum_value_loss / max(n_steps, 1),
            entropy=sum_entropy / max(n_steps, 1),
            approx_kl=sum_approx_kl / max(n_steps, 1),
            clip_frac=sum_clip_frac / max(n_steps, 1),
            explained_variance=float(ev),
            grad_norm=sum_grad_norm / max(n_steps, 1),
            lr=lr,
        )

    # ------------------------------------------------------------------
    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Return optimizer + module state for checkpointing."""
        return {
            "module": self.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "current_update": self.current_update,
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        """Restore optimizer + module state from a checkpoint."""
        self.module.load_state_dict(state["module"])  # type: ignore[arg-type]
        self.optimizer.load_state_dict(state["optimizer"])  # type: ignore[arg-type]
        self.current_update = int(state.get("current_update", 0))


def _explained_variance(y: torch.Tensor, pred: torch.Tensor) -> float:
    """Compute the fraction of variance in ``y`` explained by ``pred``."""
    var_y = y.var().item()
    if var_y < 1e-12:
        return 0.0
    return float(1.0 - (y - pred).var().item() / var_y)
