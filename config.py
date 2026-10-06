"""Central configuration for the LLM-TOC system.

All hyperparameters, buffer sizes, thresholds, and model identifiers
live here as typed dataclasses. No magic numbers should appear in any
other module - everything must be sourced from these dataclasses.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import torch


# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------
@dataclass
class DeviceConfig:
    """Device placement configuration."""

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Dedicated device for the frozen LLM (can be CPU to leave GPU free
    # for the RL policy/value networks).
    llm_device: str = "cuda" if torch.cuda.is_available() else "cpu"
    text_encoder_device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def torch_device(self) -> torch.device:
        """Return the primary torch device for RL networks."""
        return torch.device(self.device)

    def llm_torch_device(self) -> torch.device:
        """Return the device for the frozen LLM."""
        return torch.device(self.llm_device)

    def text_encoder_torch_device(self) -> torch.device:
        """Return the device for the frozen text encoder."""
        return torch.device(self.text_encoder_device)


# ---------------------------------------------------------------------------
# PPO / MARL hyperparameters (strictly enforced)
# ---------------------------------------------------------------------------
@dataclass
class PPOConfig:
    """Strict PPO hyperparameters per the project specification.

    Attributes:
        lr: Learning rate (with linear decay applied across updates).
        clip_eps: PPO clipping threshold epsilon.
        ent_coef: Entropy bonus coefficient.
        gamma: Discount factor.
        gae_lambda: Generalized Advantage Estimation parameter.
        ppo_epochs: Number of optimization epochs per rollout.
        max_grad_norm: Maximum gradient norm for clipping.
        value_coef: Value-function loss coefficient.
        target_kl: Optional early-stopping KL divergence.
        anneal_lr: Whether to linearly decay the learning rate.
        num_minibatches: Minibatch count per PPO epoch.
        normalize_advantage: Whether to standardize advantages per batch.
    """

    lr: float = 3e-4
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_epochs: int = 10
    max_grad_norm: float = 0.5
    value_coef: float = 0.5
    target_kl: float = float("inf")
    anneal_lr: bool = True
    num_minibatches: int = 4
    normalize_advantage: bool = True


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    """Top-level training loop configuration.

    Attributes:
        seeds: Random seeds for multi-seed runs (shaded convergence curves).
        total_timesteps: Total environment timesteps per seed.
        num_envs: Number of parallel environments per agent group.
        num_steps: Rollout horizon (steps per environment per rollout).
        eval_interval: Evaluation cadence in timesteps.
        eval_episodes: Number of deterministic evaluation episodes.
        save_dir: Root directory for checkpoints and logs.
        algorithm: Either "ippo" or "mappo".
        num_agents: Number of controllable agents in the environment.
    """

    seeds: Tuple[int, ...] = (42, 100, 2026)
    total_timesteps: int = 1_000_000
    num_envs: int = 4
    num_steps: int = 128
    eval_interval: int = 25_000
    eval_episodes: int = 10
    save_dir: Path = Path("./runs")
    algorithm: str = "mappo"
    num_agents: int = 2

    @property
    def batch_size(self) -> int:
        """Total transitions collected per rollout."""
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        """Per-minibatch transition count (must divide batch_size)."""
        return self.batch_size // 4

    @property
    def num_iterations(self) -> int:
        """Total PPO update iterations."""
        return self.total_timesteps // self.batch_size


# ---------------------------------------------------------------------------
# Environment / Observation
# ---------------------------------------------------------------------------
@dataclass
class EnvConfig:
    """Habitat environment and observation-wrapper configuration.

    Attributes:
        env_id: Habitat task identifier.
        scene_dataset: HSSD scene dataset path or config.
        max_episode_steps: Maximum steps per episode.
        mission_embed_dim: Dimension of the frozen text-encoder mission vector.
        agent_state_dim: Dimension of the per-agent proprioceptive state.
        coagent_rel_dim: Dimension of the relative co-agent state.
        object_dim: Dimension of each object descriptor (relative pos + target).
        max_objects: Maximum number of unplaced objects encoded in the vector.
        cache_mission_on_no_trigger: If True, reuse the last mission embedding
            when the TD-error trigger does not fire.
    """

    env_id: str = "SocialRearrangement-v0"
    scene_dataset: str = "hssd-hab200"
    max_episode_steps: int = 1000
    mission_embed_dim: int = 384
    agent_state_dim: int = 11  # pos(3) + quat(4) + ee(3) + gripper(1)
    coagent_rel_dim: int = 7  # rel pos(3) + rel quat(4)
    object_dim: int = 6  # rel pos(3) + rel target(3)
    max_objects: int = 8
    cache_mission_on_no_trigger: bool = True
    baseline_mode: bool = False

    @property
    def total_obs_dim(self) -> int:
        """Flattened vector observation dimension (excluding mission vector)."""
        return (
            self.agent_state_dim
            + self.coagent_rel_dim
            + self.max_objects * self.object_dim
        )

    @property
    def obs_with_mission_dim(self) -> int:
        """Full observation dimension including the mission embedding.
        
        When baseline_mode is True, returns total_obs_dim (no mission vector).
        """
        if self.baseline_mode:
            return self.total_obs_dim
        return self.total_obs_dim + self.mission_embed_dim


# ---------------------------------------------------------------------------
# Fast-Slow Hierarchical Engine
# ---------------------------------------------------------------------------
@dataclass
class HierarchicalConfig:
    """Configuration for the asynchronous fast-slow hierarchical engine.

    Attributes:
        trigger_threshold_init: Initial dynamic threshold theta for |delta_t|.
        trigger_threshold_scale: Multiplicative factor applied to a rolling
            statistics-based threshold (e.g., relative to std of TD errors).
        trigger_use_ema_stats: Whether to use EMA mean/std of |delta_t| to
            build a dynamic threshold (robust to reward-scale drift).
        trigger_ema_alpha: EMA smoothing factor for TD-error statistics.
        trigger_min_threshold: Lower bound on the dynamic threshold.
        top_k_salient: Number of top salient indices to retain per trigger.
        saliency_eps: Small epsilon added for numerical stability when
            normalizing saliency magnitudes.
        llm_model_id: HuggingFace identifier of the frozen LLM.
        llm_lora_path: Optional path to LoRA adapter weights.
        llm_max_new_tokens: Maximum generated tokens per LLM call.
        llm_dtype: Torch dtype used for LLM inference.
        text_encoder_id: HuggingFace identifier of the frozen text encoder.
        text_encoder_dim: Output embedding dimension of the text encoder.
        mission_cache_max_age: Maximum number of steps a cached mission
            embedding may be reused before a forced refresh.
    """

    trigger_threshold_init: float = 0.5
    trigger_threshold_scale: float = 2.0
    trigger_use_ema_stats: bool = True
    trigger_ema_alpha: float = 0.01
    trigger_min_threshold: float = 1e-3
    top_k_salient: int = 5
    saliency_eps: float = 1e-8
    llm_model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    llm_lora_path: str = ""
    llm_max_new_tokens: int = 64
    llm_dtype: str = "float16"
    text_encoder_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    text_encoder_dim: int = 384
    mission_cache_max_age: int = 50

    def llm_torch_dtype(self) -> torch.dtype:
        """Resolve the configured LLM dtype to a torch.dtype."""
        return getattr(torch, self.llm_dtype, torch.float16)


# ---------------------------------------------------------------------------
# Network architecture
# ---------------------------------------------------------------------------
@dataclass
class NetworkConfig:
    """Actor-Critic network architecture configuration.

    Attributes:
        hidden_dims: Hidden layer widths for both actor and critic trunks.
        activation: Activation function identifier ("tanh" or "relu").
        critic_shared_backbone: If True, MAPPO shares the first layers
            between all agents' critics (centralized value functions).
        init_scale: Standard deviation for orthogonal weight init.
    """

    hidden_dims: Tuple[int, ...] = (256, 256)
    activation: str = "tanh"
    critic_shared_backbone: bool = True
    init_scale: float = 0.5


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
@dataclass
class LogConfig:
    """Logging configuration (CSV + TensorBoard)."""

    csv_path: Path = Path("./runs/metrics.csv")
    tensorboard_dir: Path = Path("./runs/tb")
    log_interval_steps: int = 1
    flush_every: int = 50


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
@dataclass
class ExperimentConfig:
    """Aggregate configuration bundling every subsystem config.

    A single instance is passed around the codebase. Sub-configs are
    mutable to allow CLI overrides in train.py (e.g. baseline_mode,
    algorithm, device).
    """

    device: DeviceConfig = field(default_factory=DeviceConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    hierarchical: HierarchicalConfig = field(default_factory=HierarchicalConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    log: LogConfig = field(default_factory=LogConfig)
    experiment_name: str = "llm_toc_mappo"

    def resolve_paths(self, base: str | os.PathLike[str] | None = None) -> None:
        """Resolve all relative paths against a base directory.

        Args:
            base: Base directory. If None, uses ``save_dir``.
        """
        root = Path(base) if base is not None else self.train.save_dir
        self.train.save_dir = root
        self.log.csv_path = root / "metrics.csv"
        self.log.tensorboard_dir = root / "tb"
