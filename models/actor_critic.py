"""Actor-Critic networks for IPPO / MAPPO.

Both IPPO and MAPPO share the same network code; the only architectural
difference is whether the critic receives additional centralized state
(decentralized execution with centralized training). That distinction is
handled at the trainer level by feeding a different observation tensor to
the critic - the network classes here are observation-agnostic MLPs with
orthogonal weight initialization.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

from llm_toc.config import EnvConfig, NetworkConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _activation(name: str) -> nn.Module:
    """Resolve an activation name to a module."""
    name = name.lower()
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unknown activation: {name}")


def _build_mlp(
    in_dim: int,
    hidden_dims: Sequence[int],
    out_dim: int,
    activation: str,
    init_scale: float,
) -> nn.Sequential:
    """Build an MLP with orthogonal initialization scaled by ``init_scale``.

    Orthogonal init with adjustable gain is the standard for PPO and
    produces stable policy-value learning.
    """
    layers: list[nn.Module] = []
    last = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(last, h))
        layers.append(_activation(activation))
        last = h
    head = nn.Linear(last, out_dim)
    # Initialize body with tanh-gain; head with the requested scale.
    for m in layers:
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain(
                "tanh" if activation == "tanh" else "relu"))
            nn.init.zeros_(m.bias)
    nn.init.orthogonal_(head.weight, gain=init_scale)
    nn.init.zeros_(head.bias)
    layers.append(head)
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Actor (policy)
# ---------------------------------------------------------------------------
class Actor(nn.Module):
    """Gaussian policy for continuous control with a fixed action dim.

    For Social Rearrangement the per-agent action is typically a continuous
    base-velocity / joint command. When a discrete action space is needed
    (e.g., the stub environment), wrap this actor's mean in an argmax at
    action-selection time - the trainer handles that mapping.

    The policy outputs a mean and a state-independent learnable log-std
    (a common PPO parameterization that stabilizes training).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        net_cfg: NetworkConfig,
    ) -> None:
        """Initialize the actor.

        Args:
            obs_dim: Dimension of the (flat, mission-conditioned) obs.
            action_dim: Dimensionality of the continuous action.
            net_cfg: Network architecture configuration.
        """
        super().__init__()
        self.action_dim = action_dim
        self.mean_net = _build_mlp(
            in_dim=obs_dim,
            hidden_dims=net_cfg.hidden_dims,
            out_dim=action_dim,
            activation=net_cfg.activation,
            init_scale=0.01,  # Small init for the policy head.
        )
        # State-independent log-std parameter (PPO default).
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mean, std)`` for the Gaussian policy.

        Args:
            obs: ``(B, obs_dim)`` observation tensor.

        Returns:
            Tuple of mean and std tensors, each ``(B, action_dim)``.
        """
        mean = self.mean_net(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std

    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action and return (action, log_prob, entropy).

        Args:
            obs: ``(B, obs_dim)`` observation tensor.
            deterministic: If True, return the mean action (eval mode).

        Returns:
            Tuple ``(action, log_prob, entropy)`` each ``(B, action_dim)``
            or ``(B,)`` for the scalars.
        """
        mean, std = self.forward(obs)
        if deterministic:
            action = mean
            log_prob = torch.zeros(action.shape[0], device=obs.device)
            entropy = torch.zeros(action.shape[0], device=obs.device)
            return action, log_prob, entropy
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log-prob and entropy of given actions.

        Used during PPO updates.

        Args:
            obs: ``(B, obs_dim)`` observation tensor.
            actions: ``(B, action_dim)`` action tensor.

        Returns:
            Tuple ``(log_prob, entropy)`` each ``(B,)``.
        """
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


# ---------------------------------------------------------------------------
# Critic (value function)
# ---------------------------------------------------------------------------
class Critic(nn.Module):
    """Value function V(s) approximator.

    For MAPPO the critic may receive a *centralized* observation (e.g.,
    the concatenated state of all agents), in which case ``obs_dim``
    should be set accordingly by the trainer. The critic itself does not
    care whether its input is decentralized or centralized.
    """

    def __init__(
        self,
        obs_dim: int,
        net_cfg: NetworkConfig,
        init_scale: float = 1.0,
    ) -> None:
        """Initialize the critic.

        Args:
            obs_dim: Dimension of the value-function input.
            net_cfg: Network architecture configuration.
            init_scale: Gain for the orthogonal init of the value head.
        """
        super().__init__()
        self.value_net = _build_mlp(
            in_dim=obs_dim,
            hidden_dims=net_cfg.hidden_dims,
            out_dim=1,
            activation=net_cfg.activation,
            init_scale=init_scale,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return V(s) as a ``(B,)`` tensor."""
        return self.value_net(obs).squeeze(-1)


# ---------------------------------------------------------------------------
# Combined module
# ---------------------------------------------------------------------------
class ActorCritic(nn.Module):
    """Combined actor-critic module sharing nothing but the device.

    Keeping the actor and critic as separate trunks is the standard PPO
    recipe and avoids destructive interference between policy and value
    gradients.
    """

    def __init__(
        self,
        actor_obs_dim: int,
        critic_obs_dim: int,
        action_dim: int,
        net_cfg: NetworkConfig,
    ) -> None:
        """Initialize the actor-critic.

        Args:
            actor_obs_dim: Dimension of the actor (decentralized) input.
            critic_obs_dim: Dimension of the critic input. For IPPO this
                equals ``actor_obs_dim``; for MAPPO it equals the
                centralized state dimension.
            action_dim: Continuous action dimensionality per agent.
            net_cfg: Network architecture configuration.
        """
        super().__init__()
        self.actor = Actor(actor_obs_dim, action_dim, net_cfg)
        self.critic = Critic(critic_obs_dim, net_cfg)

    @property
    def actor_obs_dim(self) -> int:
        """Decentralized observation dimension."""
        return self.actor.mean_net[0].in_features

    @property
    def critic_obs_dim(self) -> int:
        """Critic (possibly centralized) observation dimension."""
        return self.critic.value_net[0].in_features

    @property
    def action_dim(self) -> int:
        """Action dimensionality."""
        return self.actor.action_dim


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_actor_critic(
    env_cfg: EnvConfig,
    net_cfg: NetworkConfig,
    action_dim: int,
    centralized_critic: bool = False,
    num_agents: int = 2,
) -> ActorCritic:
    """Construct an :class:`ActorCritic` for the configured environment.

    Args:
        env_cfg: Environment configuration.
        net_cfg: Network architecture configuration.
        action_dim: Continuous action dimensionality per agent.
        centralized_critic: If True (MAPPO), the critic receives the
            concatenation of all agents' observations.
        num_agents: Number of agents (only used for the centralized dim).

    Returns:
        An :class:`ActorCritic` module.
    """
    actor_dim = env_cfg.obs_with_mission_dim
    if centralized_critic:
        critic_dim = env_cfg.obs_with_mission_dim * num_agents
    else:
        critic_dim = env_cfg.obs_with_mission_dim
    return ActorCritic(
        actor_obs_dim=actor_dim,
        critic_obs_dim=critic_dim,
        action_dim=action_dim,
        net_cfg=net_cfg,
    )
