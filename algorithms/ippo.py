"""Independent PPO (IPPO) trainer.

IPPO trains each agent with its own independent actor *and* critic. The
critic receives only the agent's own decentralized observation. There is
no parameter sharing between agents and no centralized state. This is the
simplest multi-agent PPO baseline and is often competitive with MAPPO on
cooperative tasks.
"""

from __future__ import annotations

from typing import List

import torch

from llm_toc.algorithms.base_trainer import AgentHandle, BasePPOTrainer
from llm_toc.config import ExperimentConfig
from llm_toc.env.habitat_wrapper import HabitatVectorEnv
from llm_toc.models.actor_critic import build_actor_critic


class IPPOTrainer(BasePPOTrainer):
    """Independent PPO with per-agent decentralized critics."""

    algorithm_name: str = "ippo"

    def __init__(
        self,
        cfg: ExperimentConfig,
        seed: int,
        action_dim: int,
        vec_env: HabitatVectorEnv,
        logger=None,
    ) -> None:
        """Initialize IPPO. See :class:`BasePPOTrainer` for arguments."""
        super().__init__(cfg, seed, action_dim, vec_env, logger)

    # ------------------------------------------------------------------
    def _build_agents(self) -> List[AgentHandle]:
        """Build one decentralized actor-critic per agent."""
        from llm_toc.algorithms.buffer import RolloutBuffer
        from llm_toc.algorithms.ppo import PPOUpdater

        obs_dim = self.cfg.env.obs_with_mission_dim
        handles: List[AgentHandle] = []
        for _ in range(self.num_agents):
            module = build_actor_critic(
                env_cfg=self.cfg.env,
                net_cfg=self.cfg.network,
                action_dim=self.action_dim,
                centralized_critic=False,
                num_agents=self.num_agents,
            ).to(self.device)
            buffer = RolloutBuffer(
                num_steps=self.cfg.train.num_steps,
                num_envs=self.num_envs,
                actor_obs_dim=obs_dim,
                critic_obs_dim=obs_dim,
                action_dim=self.action_dim,
                device=self.device,
            )
            updater = PPOUpdater(
                module=module,
                cfg=self.cfg.ppo,
                total_updates=self.cfg.train.num_iterations,
                device=self.device,
            )
            handles.append(AgentHandle(module, buffer, updater))
        return handles

    # ------------------------------------------------------------------
    def _centralize_critic_obs(
        self, per_agent_obs: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Identity mapping: each critic receives its own decentralized obs."""
        return [obs.to(self.device).float() for obs in per_agent_obs]
