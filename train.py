"""Main training entry point for the LLM-TOC MARL system.

Usage::

    python -m llm_toc.train --algorithm mappo --seeds 42 100 2026
    python -m llm_toc.train --algorithm ippo  --seeds 42

The script:

1. Builds an :class:`ExperimentConfig` (overridable via CLI).
2. For each seed, constructs a :class:`HabitatVectorEnv` wired to a
   :class:`HierarchicalMissionManager` (the ``mission_provider``).
3. Trains with IPPO or MAPPO for ``total_timesteps`` steps.
4. Logs Collective Return and Success Rate to CSV/TensorBoard for
   shaded multi-seed convergence curves.

Run ``python -m llm_toc.train --help`` for all options.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import List, Optional, Sequence

import torch

from llm_toc.algorithms.ippo import IPPOTrainer
from llm_toc.algorithms.mappo import MAPPOTrainer
from llm_toc.algorithms.base_trainer import make_action_dim
from llm_toc.config import ExperimentConfig
from llm_toc.env.env_factory import make_vec_env
from llm_toc.utils.logger import Logger
from llm_toc.utils.seeding import set_seed

# Conditional imports for hierarchical components (only needed if enabled)
_FrozenLLMPipeline = None
_HierarchicalMissionManager = None
_FrozenTextEncoder = None

def _import_hierarchical() -> None:
    """Lazy-import hierarchical modules to avoid dependency errors in baseline mode."""
    global _FrozenLLMPipeline, _HierarchicalMissionManager, _FrozenTextEncoder
    from llm_toc.hierarchical.llm_pipeline import (
        FrozenLLMPipeline as _FL,
        HierarchicalMissionManager as _HMM,
    )
    from llm_toc.models.encoders import FrozenTextEncoder as _FTE
    _FrozenLLMPipeline = _FL
    _HierarchicalMissionManager = _HMM
    _FrozenTextEncoder = _FTE


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="LLM-TOC hierarchical MARL trainer (IPPO / MAPPO).")
    p.add_argument("--algorithm", choices=["ippo", "mappo"], default="mappo",
                   help="MARL algorithm (default: mappo).")
    p.add_argument("--seeds", type=int, nargs="+",
                   default=[42, 100, 2026],
                   help="Random seeds for multi-seed runs.")
    p.add_argument("--total-timesteps", type=int, default=1_000_000,
                   help="Total environment timesteps per seed.")
    p.add_argument("--num-envs", type=int, default=4,
                   help="Parallel environments per agent group.")
    p.add_argument("--num-steps", type=int, default=128,
                   help="Rollout horizon (steps per env per rollout).")
    p.add_argument("--save-dir", type=str, default="./runs",
                   help="Root output directory.")
    p.add_argument("--experiment-name", type=str, default="llm_toc",
                   help="Experiment name (used in run_name).")
    p.add_argument("--no-hierarchical", action="store_true",
                   help="Disable the hierarchical LLM engine (pure baseline).")
    p.add_argument("--device", type=str, default=None,
                   help="Torch device override (e.g. 'cuda' or 'cpu').")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------
def _build_config(args: argparse.Namespace) -> ExperimentConfig:
    """Build an :class:`ExperimentConfig` from CLI args."""
    cfg = ExperimentConfig()
    cfg.experiment_name = args.experiment_name
    cfg.train.algorithm = args.algorithm
    cfg.train.seeds = tuple(args.seeds)
    cfg.train.total_timesteps = args.total_timesteps
    cfg.train.num_envs = args.num_envs
    cfg.train.num_steps = args.num_steps
    cfg.train.save_dir = Path(args.save_dir)
    # Enable baseline_mode when --no-hierarchical is set.
    # This excludes the 384-dim E_mission from the observation vector,
    # ensuring a pure vector-obs baseline without any LLM interference.
    cfg.env.baseline_mode = args.no_hierarchical
    if args.device:
        cfg.device.device = args.device
        cfg.device.llm_device = args.device
        cfg.device.text_encoder_device = args.device
    cfg.resolve_paths()
    return cfg


# ---------------------------------------------------------------------------
# Mission manager factory
# ---------------------------------------------------------------------------
def _build_mission_manager(
    cfg: ExperimentConfig,
) -> Optional[object]:
    """Construct the hierarchical mission manager (or None for baseline)."""
    global _FrozenLLMPipeline, _HierarchicalMissionManager, _FrozenTextEncoder
    if _FrozenLLMPipeline is None:
        _import_hierarchical()
    # The critic is attached later by the trainer (per-agent).
    text_encoder = _FrozenTextEncoder(
        model_id=cfg.hierarchical.text_encoder_id,
        device=cfg.device.text_encoder_torch_device(),
        expected_dim=cfg.hierarchical.text_encoder_dim,
    )
    llm = _FrozenLLMPipeline(cfg.hierarchical, device=cfg.device.llm_torch_device())
    return _HierarchicalMissionManager(
        hier_cfg=cfg.hierarchical,
        env_cfg=cfg.env,
        critic=None,
        text_encoder=text_encoder,
        llm=llm,
        device=cfg.device.torch_device(),
        gamma=cfg.ppo.gamma,
    )


# ---------------------------------------------------------------------------
# Single-seed run
# ---------------------------------------------------------------------------
def _run_seed(cfg: ExperimentConfig, seed: int,
              use_hierarchical: bool) -> None:
    """Run a single training trajectory for one seed."""
    device = cfg.device.torch_device()
    set_seed(seed, device=device)

    manager: Optional[object] = None
    mission_provider = None
    if use_hierarchical:
        manager = _build_mission_manager(cfg)
        mission_provider = manager.get_mission

    vec_env = make_vec_env(
        cfg=cfg.env,
        num_envs=cfg.train.num_envs,
        base_seed=seed,
        mission_provider=mission_provider,
        device=device,
    )

    # Infer action dim from the wrapped env's underlying action space.
    action_dim = make_action_dim(vec_env.envs[0].action_space)

    logger = Logger(
        csv_path=cfg.log.csv_path,
        tensorboard_dir=cfg.log.tensorboard_dir,
        run_name=f"{cfg.experiment_name}_{cfg.train.algorithm}_seed{seed}",
    )

    if cfg.train.algorithm == "ippo":
        trainer = IPPOTrainer(cfg, seed, action_dim, vec_env, logger=logger)
    else:
        trainer = MAPPOTrainer(cfg, seed, action_dim, vec_env, logger=logger)

    # Wire the hierarchical manager to the first agent's critic (for
    # saliency). The manager caches missions per env wrapper.
    if manager is not None and trainer.agents:
        manager.attach_critic(trainer.agents[0].module.critic)

    try:
        trainer.train()
    finally:
        trainer.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    """Script entry point.

    Returns:
        Process exit code (0 on success).
    """
    args = _parse_args(argv)
    cfg = _build_config(args)
    use_hierarchical = not args.no_hierarchical

    print("=" * 72)
    print(f"LLM-TOC MARL: algorithm={cfg.train.algorithm} "
          f"hierarchical={use_hierarchical} seeds={cfg.train.seeds}")
    print(f"Device: {cfg.device.device}")
    print(f"Output: {cfg.train.save_dir.resolve()}")
    print("=" * 72)

    for seed in cfg.train.seeds:
        print(f"\n=== Seed {seed} ===")
        try:
            _run_seed(cfg, seed, use_hierarchical)
        except KeyboardInterrupt:
            print(f"Interrupted during seed {seed}.")
            return 130
        except Exception:  # noqa: BLE001
            print(f"Error during seed {seed}:")
            traceback.print_exc()
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
