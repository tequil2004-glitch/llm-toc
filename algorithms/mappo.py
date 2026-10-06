"""Multi-Agent PPO (MAPPO) trainer.

MAPPO trains decentralized actors with a *centralized* critic: the
critic for every agent receives the concatenation of all agents'
observations. This implements the CTDE (Centralized Training,
Decentralized Execution) paradigm and typically outperforms IPPO on
cooperative tasks by giving the value function full state visibility.

Implementation notes:
    * Each agent still owns its own critic instance (no parameter sharing
      between agents by default - matching the original MAPPO paper).
    * The centralized observation is built by concatenating the per-agent
      flat vectors along the feature dimension. A shared critic backbone
      (``NetworkConfig.critic_shared_backbone``) could be enabled in the
      future without changing this trainer.
"""

from __future__ import annotations

from typing import List

import torch

from llm_toc.algorithms.base_trainer import AgentHandle, BasePPOTrainer
from llm_toc.config import ExperimentConfig
from llm_toc.env.habitat_wrapper import HabitatVectorEnv
from llm_toc.models.actor_critic import build_actor_critic


class MAPPOTrainer(BasePPOTrainer):
    """MAPPO with decentralized actors and centralized critics (CTDE)."""

    algorithm_name: str = "mappo"

    def __init__(
        self,
        cfg: ExperimentConfig,
        seed: int,
        action_dim: int,
        vec_env: HabitatVectorEnv,
        logger=None,
    ) -> None:
        """Initialize MAPPO. See :class:`BasePPOTrainer` for arguments."""
        super().__init__(cfg, seed, action_dim, vec_env, logger)

    # ------------------------------------------------------------------
    def _build_agents(self) -> List[AgentHandle]:
        """Build per-agent actor-critics with centralized critic inputs."""
        from llm_toc.algorithms.buffer import RolloutBuffer
        from llm_toc.algorithms.ppo import PPOUpdater

        actor_obs_dim = self.cfg.env.obs_with_mission_dim
        critic_obs_dim = actor_obs_dim * self.num_agents  # centralized
        handles: List[AgentHandle] = []
        for _ in range(self.num_agents):
            module = build_actor_critic(
                env_cfg=self.cfg.env,
                net_cfg=self.cfg.network,
                action_dim=self.action_dim,
                centralized_critic=True,
                num_agents=self.num_agents,
            ).to(self.device)
            buffer = RolloutBuffer(
                num_steps=self.cfg.train.num_steps,
                num_envs=self.num_envs,
                actor_obs_dim=actor_obs_dim,
                critic_obs_dim=critic_obs_dim,
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
        """Concatenate all agents' obs and broadcast back per agent.

        Each agent's critic sees the *same* centralized state
        ``(num_envs, num_agents * obs_dim)``.
        """
        stacked = torch.cat(
            [obs.to(self.device).float() for obs in per_agent_obs],
            dim=-1,
        )
        # Broadcast the same centralized tensor to each agent's critic.
        return [stacked for _ in range(self.num_agents)]
