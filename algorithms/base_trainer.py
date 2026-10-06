"""Shared base trainer for IPPO and MAPPO.

Both algorithms differ only in (a) whether the critic is centralized and
(b) how rollout data is gathered per agent. The orchestration of the
training loop, action selection, logging, and checkpointing is identical
and lives here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from llm_toc.algorithms.buffer import RolloutBatch, RolloutBuffer
from llm_toc.algorithms.ppo import PPOUpdateStats, PPOUpdater
from llm_toc.config import ExperimentConfig
from llm_toc.env.habitat_wrapper import HabitatVectorEnv
from llm_toc.models.actor_critic import ActorCritic, build_actor_critic
from llm_toc.utils.logger import Logger
from llm_toc.utils.seeding import set_seed


# ---------------------------------------------------------------------------
# Per-agent container
# ---------------------------------------------------------------------------
@dataclass
class AgentHandle:
    """Container for one agent's networks, buffer, and PPO updater."""

    module: ActorCritic
    buffer: RolloutBuffer
    updater: PPOUpdater


# ---------------------------------------------------------------------------
# Episode rollouts stats
# ---------------------------------------------------------------------------
@dataclass
class EpisodeStats:
    """Aggregated episode statistics emitted at the end of each rollout."""

    collective_return: float
    success_rate: float
    episode_length: float
    num_episodes: int


# ---------------------------------------------------------------------------
# Base trainer
# ---------------------------------------------------------------------------
class BasePPOTrainer:
    """Base class implementing the shared MARL PPO training loop.

    Subclasses must implement :meth:`_build_agents` (constructing the
    per-agent :class:`AgentHandle` list, including the choice of
    centralized vs. decentralized critic) and
    :meth:`_centralize_critic_obs` (turning a list of per-env, per-agent
    actor observations into the critic observation tensor).

    The trainer integrates with:
        * :class:`HabitatVectorEnv` for the environment.
        * :class:`Logger` for CSV + TensorBoard logging.
        * :class:`set_seed` for reproducibility across the three seeds.
    """

    algorithm_name: str = "base"

    def __init__(
        self,
        cfg: ExperimentConfig,
        seed: int,
        action_dim: int,
        vec_env: HabitatVectorEnv,
        logger: Optional[Logger] = None,
    ) -> None:
        """Initialize the trainer.

        Args:
            cfg: Aggregate experiment configuration.
            seed: RNG seed for this run.
            action_dim: Continuous action dimensionality per agent.
            vec_env: Vectorized environment.
            logger: Optional logger; a new one is created if None.
        """
        self.cfg = cfg
        self.seed = seed
        self.action_dim = action_dim
        self.vec_env = vec_env
        self.device = cfg.device.torch_device()
        self.num_agents = vec_env.num_agents
        self.num_envs = vec_env.num_envs

        set_seed(seed, device=self.device)
        self.rng = np.random.default_rng(seed)

        self.logger = logger or Logger(
            csv_path=cfg.log.csv_path,
            tensorboard_dir=cfg.log.tensorboard_dir,
            run_name=f"{cfg.experiment_name}_{self.algorithm_name}_seed{seed}",
        )

        self.agents: List[AgentHandle] = self._build_agents()
        self.global_step = 0
        self.start_time = time.time()
        self._episode_returns: List[List[float]] = [
            [0.0] * self.num_envs for _ in range(self.num_agents)]
        self._episode_lengths: List[List[int]] = [
            [0] * self.num_envs for _ in range(self.num_agents)]
        self._episode_success: List[List[bool]] = [
            [False] * self.num_envs for _ in range(self.num_agents)]
        self._completed_returns: List[float] = []
        self._completed_successes: List[bool] = []
        self._completed_lengths: List[int] = []

    # ------------------------------------------------------------------
    # To be implemented by subclasses
    # ------------------------------------------------------------------
    def _build_agents(self) -> List[AgentHandle]:
        """Construct per-agent modules, buffers, and updaters."""
        raise NotImplementedError

    def _centralize_critic_obs(
        self, per_agent_obs: List[torch.Tensor]
    ) -> torch.Tensor:
        """Map per-agent decentralized obs to per-agent critic obs.

        For IPPO this is the identity. For MAPPO it concatenates all
        agents' obs (per environment) and broadcasts back to per-agent
        views.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _act(
        self,
        obs_lists: List[List[torch.Tensor]],
        deterministic: bool = False,
    ) -> Tuple[List[List[np.ndarray]], List[List[float]], List[List[float]]]:
        """Sample actions for all agents across all envs.

        Args:
            obs_lists: ``obs_lists[env][agent]`` is a 1-D obs tensor.
            deterministic: If True, use the mean action (eval mode).

        Returns:
            Tuple ``(actions, log_probs, values)`` each shaped
            ``[env][agent]``. ``actions`` are NumPy arrays.
        """
        actions_out: List[List[np.ndarray]] = []
        log_probs_out: List[List[float]] = []
        values_out: List[List[float]] = []

        # Stack per-agent obs across envs.
        # per_agent_obs[agent] = (num_envs, obs_dim)
        per_agent_obs: List[torch.Tensor] = [
            torch.stack([obs_lists[e][a] for e in range(self.num_envs)],
                        dim=0).to(self.device).float()
            for a in range(self.num_agents)
        ]

        critic_obs = self._centralize_critic_obs(per_agent_obs)

        for a, handle in enumerate(self.agents):
            actor_obs = per_agent_obs[a]
            critic_obs_a = critic_obs[a] if isinstance(critic_obs, list) else critic_obs
            action, log_prob, _ = handle.module.actor.get_action(
                actor_obs, deterministic=deterministic)
            value = handle.module.critic(critic_obs_a)
            actions_out.append([ai.detach().cpu().numpy() for ai in action])
            log_probs_out.append([float(lp) for lp in log_prob.cpu().numpy()])
            values_out.append([float(v) for v in value.cpu().numpy()])

        # Reshape to [env][agent].
        out_actions: List[List[np.ndarray]] = [
            [actions_out[a][e] for a in range(self.num_agents)]
            for e in range(self.num_envs)
        ]
        out_log_probs: List[List[float]] = [
            [log_probs_out[a][e] for a in range(self.num_agents)]
            for e in range(self.num_envs)
        ]
        out_values: List[List[float]] = [
            [values_out[a][e] for a in range(self.num_agents)]
            for e in range(self.num_envs)
        ]
        return out_actions, out_log_probs, out_values

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------
    def _collect_rollout(self) -> None:
        """Collect ``num_steps`` of experience into per-agent buffers."""
        cfg = self.cfg.train
        obs_lists = self._last_obs

        for handle in self.agents:
            handle.buffer.reset()

        for t in range(cfg.num_steps):
            actions, log_probs, values = self._act(obs_lists)
            next_obs, rewards, dones, successes, info = self.vec_env.step(
                actions)

            # Use collective reward for each agent (team reward).
            # Build per-agent actor obs and (possibly centralized) critic obs
            # once per timestep - hoisted out of the per-agent loop because
            # the centralized critic obs is shared across agents.
            per_agent_obs = [
                torch.stack([obs_lists[e][ai]
                             for e in range(self.num_envs)], dim=0
                            ).to(self.device).float()
                for ai in range(self.num_agents)
            ]
            critic_obs_t = self._centralize_critic_obs(per_agent_obs)

            for a, handle in enumerate(self.agents):
                actor_obs_np = per_agent_obs[a].cpu().numpy().astype(np.float32)
                critic_obs_a = (
                    critic_obs_t[a]
                    if isinstance(critic_obs_t, list) else critic_obs_t
                )
                critic_obs_np = critic_obs_a.cpu().numpy().astype(np.float32)
                actions_np = np.stack(
                    [actions[e][a] for e in range(self.num_envs)], axis=0
                ).astype(np.float32)
                lp_np = np.array(
                    [log_probs[e][a] for e in range(self.num_envs)],
                    dtype=np.float32)
                v_np = np.array(
                    [values[e][a] for e in range(self.num_envs)],
                    dtype=np.float32)
                r_np = rewards.astype(np.float32)  # collective reward
                d_np = dones.astype(np.float32)
                handle.buffer.insert(
                    actor_obs=actor_obs_np,
                    critic_obs=critic_obs_np,
                    actions=actions_np,
                    log_probs=lp_np,
                    rewards=r_np,
                    values=v_np,
                    dones=d_np,
                )

                # Episode bookkeeping.
                for e in range(self.num_envs):
                    self._episode_returns[a][e] += float(rewards[e])
                    self._episode_lengths[a][e] += 1
                    if dones[e]:
                        self._completed_returns.append(
                            self._episode_returns[a][e])
                        self._completed_successes.append(
                            bool(self._episode_success[a][e] or
                                 successes[e]))
                        self._completed_lengths.append(
                            self._episode_lengths[a][e])
                        self._episode_returns[a][e] = 0.0
                        self._episode_lengths[a][e] = 0
                        self._episode_success[a][e] = False
                    elif successes[e]:
                        self._episode_success[a][e] = True

            self.global_step += self.num_envs
            obs_lists = next_obs

        self._last_obs = obs_lists

        # Bootstrap values for GAE.
        per_agent_obs = [
            torch.stack([obs_lists[e][ai]
                         for e in range(self.num_envs)], dim=0
                        ).to(self.device).float()
            for ai in range(self.num_agents)
        ]
        critic_obs_t = self._centralize_critic_obs(per_agent_obs)
        for a, handle in enumerate(self.agents):
            critic_obs_a = (
                critic_obs_t[a]
                if isinstance(critic_obs_t, list) else critic_obs_t
            )
            with torch.no_grad():
                last_values = handle.module.critic(critic_obs_a).cpu().numpy()
            handle.buffer.compute_advantages(
                last_values=last_values,
                gamma=self.cfg.ppo.gamma,
                gae_lambda=self.cfg.ppo.gae_lambda,
                normalize=self.cfg.ppo.normalize_advantage,
            )

    # ------------------------------------------------------------------
    def _update(self) -> List[PPOUpdateStats]:
        """Run PPO updates for all agents and return per-agent stats."""
        all_stats: List[PPOUpdateStats] = []
        for handle in self.agents:
            batches = handle.buffer.get_minibatches(
                num_minibatches=self.cfg.ppo.num_minibatches,
                rng=self.rng,
            )
            stats = handle.updater.update(batches, rng=self.rng)
            all_stats.append(stats)
        return all_stats

    # ------------------------------------------------------------------
    def _episode_stats(self) -> EpisodeStats:
        """Aggregate completed-episode statistics since the last call."""
        if not self._completed_returns:
            return EpisodeStats(0.0, 0.0, 0.0, 0)
        cr = float(np.mean(self._completed_returns))
        sr = float(np.mean(self._completed_successes))
        el = float(np.mean(self._completed_lengths))
        n = len(self._completed_returns)
        self._completed_returns.clear()
        self._completed_successes.clear()
        self._completed_lengths.clear()
        return EpisodeStats(cr, sr, el, n)

    # ------------------------------------------------------------------
    def train(self) -> None:
        """Run the full training loop for the configured number of steps."""
        cfg = self.cfg.train
        self._last_obs = self.vec_env.reset()
        self.logger.log_scalar("train/seed", self.seed, step=0)
        num_iters = cfg.num_iterations

        for it in range(num_iters):
            self._collect_rollout()
            stats_list = self._update()
            ep_stats = self._episode_stats()

            sps = self.global_step / max(time.time() - self.start_time, 1e-6)
            for a, st in enumerate(stats_list):
                for k, v in st.as_dict().items():
                    self.logger.log_scalar(
                        f"train/agent{a}/{k}", v, step=self.global_step)
            self.logger.log_scalar(
                "train/collective_return", ep_stats.collective_return,
                step=self.global_step)
            self.logger.log_scalar(
                "train/success_rate", ep_stats.success_rate,
                step=self.global_step)
            self.logger.log_scalar(
                "train/episode_length", ep_stats.episode_length,
                step=self.global_step)
            self.logger.log_scalar(
                "train/episodes", ep_stats.num_episodes, step=self.global_step)
            self.logger.log_scalar(
                "train/steps_per_second", sps, step=self.global_step)
            self.logger.log_scalar(
                "train/iteration", it, step=self.global_step)

            if (it + 1) % max(1, self.cfg.log.flush_every) == 0:
                self.logger.flush()
                print(
                    f"[{self.algorithm_name} seed={self.seed}] "
                    f"iter {it+1}/{num_iters} step={self.global_step} "
                    f"CR={ep_stats.collective_return:.3f} "
                    f"SR={ep_stats.success_rate:.3f} "
                    f"sps={sps:.1f}"
                )

        self.logger.flush()
        self._save_checkpoint()

    # ------------------------------------------------------------------
    def _save_checkpoint(self) -> None:
        """Save final per-agent checkpoints."""
        ckpt_dir = self.cfg.train.save_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        for a, handle in enumerate(self.agents):
            path = ckpt_dir / f"{self.algorithm_name}_seed{self.seed}_agent{a}.pt"
            torch.save(handle.updater.state_dict(), path)

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Close the environment and logger."""
        try:
            self.vec_env.close()
        finally:
            self.logger.close()


# ---------------------------------------------------------------------------
# Helpers used by both subclasses
# ---------------------------------------------------------------------------
def make_action_dim(action_space: Any) -> int:
    """Infer the continuous action dimension from a gym/gymnasium space.

    Supports :class:`gym.spaces.Box` (returns ``shape[0]``) and
    :class:`gym.spaces.Discrete` (returns 1, treated as a continuous
    logit-index by the actor). The trainer discretizes back to a discrete
    action via ``argmax`` at env-step time.
    """
    import gymnasium as gym
    # Also handle old gym spaces (Habitat returns gym.spaces.Box).
    try:
        import gym as old_gym
        _old_box = old_gym.spaces.Box
        _old_discrete = old_gym.spaces.Discrete
        _old_multi = old_gym.spaces.MultiDiscrete
    except ImportError:
        _old_box = _old_discrete = _old_multi = None

    if isinstance(action_space, gym.spaces.Box) or (
        _old_box is not None and isinstance(action_space, _old_box)
    ):
        return int(action_space.shape[0])
    if isinstance(action_space, gym.spaces.Discrete) or (
        _old_discrete is not None and isinstance(action_space, _old_discrete)
    ):
        return 1
    if isinstance(action_space, gym.spaces.MultiDiscrete) or (
        _old_multi is not None and isinstance(action_space, _old_multi)
    ):
        return int(len(action_space.nvec))
    raise ValueError(f"Unsupported action space: {action_space}")


def build_default_logger(
    cfg: ExperimentConfig, algorithm_name: str, seed: int
) -> Logger:
    """Construct a :class:`Logger` with algorithm/seed-scoped paths."""
    return Logger(
        csv_path=cfg.log.csv_path,
        tensorboard_dir=cfg.log.tensorboard_dir,
        run_name=f"{cfg.experiment_name}_{algorithm_name}_seed{seed}",
    )
