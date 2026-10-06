"""Vector observation wrapper for Habitat-Lab Social Rearrangement.

The wrapper converts Habitat's dictionary observations into a single flat
``torch.Tensor`` per agent that concatenates:

1. Agent state (position, orientation, end-effector, gripper state).
2. Relative pose of the co-agent (humanoid avatar).
3. Relative positions of all unplaced objects and their target zones.
4. The mission embedding vector ``E_mission`` produced by the frozen text
   encoder. When no new mission has been generated, the previously cached
   vector is reused (fast path - no LLM call).

The wrapper is intentionally decoupled from the LLM pipeline: it accepts an
external ``mission_provider`` callable that returns the (possibly cached)
mission embedding. This keeps the environment wrapper unit-testable without
GPU/LLM dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import os
import numpy as np
import torch
from gymnasium import spaces

from llm_toc.config import EnvConfig


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class VectorObs:
    """Structured view of a flattened vector observation.

    Attributes:
        agent_state: Proprioceptive state of the agent.
        coagent_rel: Relative pose of the co-agent.
        objects: (max_objects, object_dim) tensor of object descriptors.
            Rows beyond the actual object count are zero-padded.
        mission: The mission embedding vector (cached if no new trigger).
        flat: The concatenated flat observation tensor consumed by the
            policy/value networks.
        valid_object_mask: Boolean mask of length ``max_objects`` indicating
            which object slots are populated by real unplaced objects.
    """

    agent_state: torch.Tensor
    coagent_rel: torch.Tensor
    objects: torch.Tensor
    mission: torch.Tensor
    valid_object_mask: torch.Tensor
    flat: torch.Tensor

    @classmethod
    def from_flat(
        cls,
        flat: torch.Tensor,
        cfg: EnvConfig,
    ) -> "VectorObs":
        """Deconstruct a flat observation vector into structured fields.

        Args:
            flat: 1-D tensor of length ``cfg.obs_with_mission_dim``.
            cfg: Environment configuration specifying field widths.

        Returns:
            A :class:`VectorObs` view sharing memory with ``flat``.
        """
        offset = 0
        agent_state = flat[offset : offset + cfg.agent_state_dim]
        offset += cfg.agent_state_dim

        coagent_rel = flat[offset : offset + cfg.coagent_rel_dim]
        offset += cfg.coagent_rel_dim

        obj_len = cfg.max_objects * cfg.object_dim
        objects = flat[offset : offset + obj_len].view(
            cfg.max_objects, cfg.object_dim
        )
        offset += obj_len

        # In baseline_mode, the observation does not include a mission embedding.
        # Return a zero vector as a placeholder for downstream consumers.
        if cfg.baseline_mode:
            mission = torch.zeros(cfg.mission_embed_dim, device=flat.device)
        else:
            mission = flat[offset : offset + cfg.mission_embed_dim]
        return cls(
            agent_state=agent_state,
            coagent_rel=coagent_rel,
            objects=objects,
            mission=mission,
            valid_object_mask=(objects.abs().sum(dim=-1) > 0),
            flat=flat,
        )


# ---------------------------------------------------------------------------
# Mission provider protocol
# ---------------------------------------------------------------------------
MissionProvider = Callable[[], torch.Tensor]
"""A callable returning the current mission embedding ``E_mission``.

The provider is responsible for caching the last LLM-generated mission
vector when the TD-error trigger does not fire. The environment wrapper
does not perform any LLM calls itself.
"""


# ---------------------------------------------------------------------------
# Helper extraction functions (Habitat-agnostic, dict-based)
# ---------------------------------------------------------------------------
def _as_quaternion_wxyz(agent_state: Dict[str, Any]) -> np.ndarray:
    """Extract agent orientation as a (4,) quaternion (w, x, y, z)."""
    rot = agent_state.get("rotation", None)
    if rot is None:
        return np.zeros(4, dtype=np.float32)
    if isinstance(rot, (list, tuple, np.ndarray)):
        arr = np.asarray(rot, dtype=np.float32).reshape(-1)
        if arr.size == 4:
            return arr
    # habitat sim Quaternion
    return np.array(
        [getattr(rot, "w", 0.0), getattr(rot, "x", 0.0),
         getattr(rot, "y", 0.0), getattr(rot, "z", 0.0)],
        dtype=np.float32,
    )


def _extract_agent_state(
    obs: Dict[str, Any], agent_id: int
) -> np.ndarray:
    """Extract per-agent proprioceptive state.

    Returns a vector of length EnvConfig.agent_state_dim (11).

    Two observation contracts are supported:

    * Multi-agent contract (SocialRearrangement-v0 / stub env):
      obs["agent_states"][agent_id] exposing position, rotation,
      end_effector and gripper -> [pos(3), quat(4), ee(3), grip(1)].
    * Habitat-Lab rearrange contract (single fixed-base Fetch arm):
      [ee_pos(3), joint(7), is_holding(1)]. The Fetch base is static in
      rearrange tasks, so base position/orientation carry no information
      and are replaced by the arm joint angles, which are the actual
      proprioceptive state of the controllable agent.
    """
    agent_states = obs.get("agent_states") or obs.get("agents")
    if agent_states is not None:
        try:
            a = agent_states[agent_id]
            pos = np.asarray(a.get("position", [0.0, 0.0, 0.0]),
                             dtype=np.float32)[:3]
            quat = _as_quaternion_wxyz(a)
            ee = np.asarray(a.get("end_effector", [0.0, 0.0, 0.0]),
                            dtype=np.float32)[:3]
            gripper = float(a.get("gripper", 0.0))
            return np.concatenate([pos, quat, ee, [gripper]]).astype(np.float32)
        except Exception:  # noqa: BLE001 - fall through to rearrange contract
            pass

    # --- Habitat-Lab rearrange: [ee_pos(3), joint(7), is_holding(1)] ---
    out = np.zeros(11, dtype=np.float32)
    ee = obs.get("ee_pos")
    if ee is not None:
        v = np.asarray(ee, dtype=np.float32).reshape(-1)
        out[0:3] = v[:3]
    joint = obs.get("joint")
    if joint is not None:
        v = np.asarray(joint, dtype=np.float32).reshape(-1)
        n = min(7, v.size)
        out[3:3 + n] = v[:n]
    holding = obs.get("is_holding")
    if holding is not None:
        v = np.asarray(holding, dtype=np.float32).reshape(-1)
        out[10] = float(v[0]) if v.size else 0.0
    return out


def _extract_coagent_rel(
    obs: Dict[str, Any], agent_id: int, num_agents: int
) -> np.ndarray:
    """Extract the relative pose of the co-agent w.r.t. ``agent_id``.

    For the canonical 2-agent case this returns a 7-vector
    ``[dx, dy, dz, dqw, dqx, dqy, dqz]``. When more agents exist, the
    first co-agent is reported. Zero-padded when there is no co-agent.
    """
    agent_states = obs.get("agent_states") or obs.get("agents")
    if agent_states is None or num_agents < 2:
        return np.zeros(7, dtype=np.float32)
    self_a = agent_states[agent_id]
    other_idx = 1 if agent_id == 0 else 0
    other = agent_states[other_idx]
    p_self = np.asarray(self_a.get("position", [0.0, 0.0, 0.0]),
                        dtype=np.float32)[:3]
    p_other = np.asarray(other.get("position", [0.0, 0.0, 0.0]),
                         dtype=np.float32)[:3]
    q_self = _as_quaternion_wxyz(self_a)
    q_other = _as_quaternion_wxyz(other)
    # Relative position in the world frame (sufficient for vector policy).
    dpos = p_other - p_self
    # Relative orientation as the component-wise delta quaternion
    # (a simplified proxy; full quaternion algebra is unnecessary here).
    dquat = q_other - q_self
    return np.concatenate([dpos, dquat]).astype(np.float32)


def _extract_objects(
    obs: Dict[str, Any],
    agent_state: np.ndarray,
    cfg: EnvConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract object descriptors.

    Each row is [rel_obj_x, rel_obj_y, rel_obj_z, rel_tgt_x, rel_tgt_y,
    rel_tgt_z]. Rows beyond max_objects are truncated; missing slots are
    zero-padded and marked invalid.

    Two contracts are supported:

    * Multi-agent / stub contract: obs["objects"] is a list of dicts with
      absolute position / target_position; they are converted to
      agent-relative coordinates.
    * Habitat-Lab rearrange contract: obj_start_sensor / obj_goal_sensor
      are already expressed relative to the end effector, so they are used
      as-is for the first object slot.

    Returns:
        Tuple of (objects, valid_mask) where objects has shape
        (max_objects, object_dim) and valid_mask has shape (max_objects,).
    """
    objects = np.zeros((cfg.max_objects, cfg.object_dim), dtype=np.float32)
    mask = np.zeros(cfg.max_objects, dtype=bool)

    objs = obs.get("objects") or obs.get("unplaced_objects") or []
    if len(objs) > 0:
        agent_pos = agent_state[:3]
        rows: List[np.ndarray] = []
        for o in objs[: cfg.max_objects]:
            op = np.asarray(o.get("position", [0.0, 0.0, 0.0]),
                            dtype=np.float32)[:3]
            tp = np.asarray(o.get("target_position",
                                  o.get("target", [0.0, 0.0, 0.0])),
                            dtype=np.float32)[:3]
            rows.append(np.concatenate([op - agent_pos,
                                        tp - agent_pos]).astype(np.float32))
        n = len(rows)
        objects[:n] = np.stack(rows, axis=0)
        mask[:n] = True
        return objects, mask

    # --- Habitat-Lab rearrange: one target object, already EE-relative ---
    start = obs.get("obj_start_sensor")
    goal = obs.get("obj_goal_sensor")
    if start is not None or goal is not None:
        s = (np.asarray(start, dtype=np.float32).reshape(-1)[:3]
             if start is not None else np.zeros(3, dtype=np.float32))
        g = (np.asarray(goal, dtype=np.float32).reshape(-1)[:3]
             if goal is not None else np.zeros(3, dtype=np.float32))
        objects[0] = np.concatenate([s, g]).astype(np.float32)
        mask[0] = True
    return objects, mask


def _suppress_terminate_action(action: Any, action_space: Any) -> Any:
    """Pin Habitat's end-episode action component to "do not terminate".

    The Habitat-Lab rearrange gym action space is ``Box(11,)`` laid out as
    ``[arm(7), grip(1), base_velocity(2), terminate(1)]``.  Probing each
    dimension with +/-1 shows that only the final one sets
    ``does_want_terminate`` / ``bad_called_terminate`` and ends the episode
    on a positive value.

    Args:
        action: Action emitted by the policy.
        action_space: The underlying env's action space (may be ``None``).

    Returns:
        The action with the terminate component pinned to the most negative
        allowed value, or the action unchanged when it cannot be identified.
    """
    if os.environ.get("LLM_TOC_DISABLE_TERMINATE_ACTION", "1") != "1":
        return action
    if action_space is None or not hasattr(action_space, "shape"):
        return action
    try:
        size = int(np.prod(action_space.shape))
    except Exception:  # noqa: BLE001
        return action
    # Only the rearrange 11-D space carries the terminate component.
    if size != 11:
        return action
    arr = np.asarray(action, dtype=np.float32).reshape(-1)
    if arr.size != 11:
        return action
    arr = arr.copy()
    arr[10] = -1.0
    return arr.reshape(np.asarray(action).shape)

# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------
class VectorObservationWrapper:
    """Wrapper producing a flat, mission-conditioned vector observation.

    This wrapper is API-compatible with both ``gym.Env`` (old 4-value step,
    single-value reset) and ``gymnasium.Env`` (5-value step, tuple reset).
    It normalizes all calls to the gymnasium API externally while delegating
    to whichever API the underlying env uses.

    The wrapper is multi-agent aware: each call to :meth:`observation`
    produces a list of per-agent flat tensors of length
    ``EnvConfig.obs_with_mission_dim``. For single-agent rollouts a single
    tensor (not a list) is returned.

    The mission embedding is sourced from an injected ``mission_provider``
    callable. When ``mission_provider`` is ``None`` or returns ``None``,
    and ``EnvConfig.cache_mission_on_no_trigger`` is True, the previous
    mission vector is reused. If no cached vector exists yet, a zero
    vector is used as a cold-start placeholder.
    """

    def __init__(
        self,
        env: Any,
        cfg: EnvConfig,
        mission_provider: Optional[MissionProvider] = None,
        num_agents: int = 2,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize the wrapper.

        Args:
            env: The underlying (Habitat) environment (gym or gymnasium).
            cfg: Environment configuration.
            mission_provider: Callable returning the current mission
                embedding. May return ``None`` to signal "use cache".
            num_agents: Number of controllable agents.
            device: Torch device for produced tensors.
        """
        self.env = env
        self.cfg = cfg
        self.mission_provider = mission_provider
        self.num_agents = num_agents
        self.device = device or torch.device("cpu")

        # Per-agent mission cache (supports independent agent views).
        self._mission_cache: List[Optional[torch.Tensor]] = [None] * num_agents

        # Build the observation space.
        # In baseline_mode, obs excludes the mission embedding (pure vector obs).
        obs_dim = cfg.obs_with_mission_dim
        low = np.full(obs_dim, -np.inf, dtype=np.float32)
        high = np.full(obs_dim, np.inf, dtype=np.float32)
        per_agent_space = spaces.Box(low=low, high=high, dtype=np.float32)
        if num_agents > 1:
            self.observation_space = spaces.Tuple(
                (per_agent_space,) * num_agents
            )
        else:
            self.observation_space = per_agent_space

        # Inherit action_space from the underlying env.
        self.action_space = getattr(env, "action_space", None)

    # ------------------------------------------------------------------
    # Mission cache management
    # ------------------------------------------------------------------
    def _resolve_mission(self, agent_id: int) -> torch.Tensor:
        """Return the current mission embedding for ``agent_id``.

        The method consults the mission provider. If the provider returns
        a valid tensor it is cached and returned. If it returns ``None``
        and caching is enabled, the previously cached vector is reused.
        A zero vector is used as a cold-start fallback.
        """
        provided: Optional[torch.Tensor] = None
        if self.mission_provider is not None:
            try:
                provided = self.mission_provider()
            except Exception:  # noqa: BLE001 - provider must never break env
                provided = None

        if provided is not None:
            provided = provided.detach().to(self.device).float()
            if provided.shape[-1] != self.cfg.mission_embed_dim:
                raise ValueError(
                    f"Mission embedding dim {provided.shape[-1]} != "
                    f"cfg.mission_embed_dim {self.cfg.mission_embed_dim}"
                )
            self._mission_cache[agent_id] = provided
            return provided

        if self.cfg.cache_mission_on_no_trigger:
            cached = self._mission_cache[agent_id]
            if cached is not None:
                return cached

        # Cold-start fallback.
        zero = torch.zeros(self.cfg.mission_embed_dim,
                           device=self.device, dtype=torch.float32)
        self._mission_cache[agent_id] = zero
        return zero

    # ------------------------------------------------------------------
    # Observation transformation
    # ------------------------------------------------------------------
    def observation(self, observation: Any) -> Any:
        """Flatten a Habitat dict observation into per-agent tensors."""
        # Idempotency guard: if we are nested inside another wrapper the
        # input is already a list of per-agent tensors; pass it through
        # instead of silently degrading to an all-zero observation.
        if isinstance(observation, (list, tuple)):
            return list(observation)

        obs = observation if isinstance(observation, dict) else {}
        flats: List[torch.Tensor] = []
        for agent_id in range(self.num_agents):
            agent_state = _extract_agent_state(obs, agent_id)
            coagent_rel = _extract_coagent_rel(obs, agent_id, self.num_agents)
            objects, _ = _extract_objects(obs, agent_state, self.cfg)

            parts = [
                torch.from_numpy(agent_state).to(self.device).float(),
                torch.from_numpy(coagent_rel).to(self.device).float(),
                torch.from_numpy(objects).to(self.device).float().view(-1),
            ]

            # In baseline_mode, exclude the mission embedding entirely.
            # This ensures pure vector obs without any LLM interference.
            if not self.cfg.baseline_mode:
                mission = self._resolve_mission(agent_id)
                parts.append(mission)

            flats.append(torch.cat(parts, dim=0))

        if self.num_agents == 1:
            return flats[0]
        return flats

    # ------------------------------------------------------------------
    # Env API (normalizes gym/gymnasium to gymnasium 5-tuple)
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None,
              options: Optional[dict] = None):
        """Reset the env and return (obs, info) in gymnasium format."""
        result = self.env.reset()
        if isinstance(result, tuple) and len(result) == 2:
            raw_obs, info = result
        else:
            raw_obs, info = result, {}
        return self.observation(raw_obs), info

    def step(self, action: Any):
        """Step the env and return gymnasium 5-tuple.

        Handles both old gym (4-value) and gymnasium (5-value) step APIs.
        For multi-agent trainers that pass per-agent action lists, only the
        first agent's action is forwarded (Habitat Social Rearrangement
        controls the Fetch robot; the humanoid is environment-managed).
        Actions are clipped to the env's action-space bounds before stepping.
        """
        if isinstance(action, (list, tuple)) and len(action) > 0 \
                and not isinstance(action, np.ndarray):
            action = action[0]
        # Clip to action space bounds to satisfy Habitat's assertion.
        if hasattr(self.env, "action_space") and \
                hasattr(self.env.action_space, "low"):
            action = np.clip(action,
                             self.env.action_space.low,
                             self.env.action_space.high)
        # Suppress the Habitat "terminate episode" action dimension.
        # In the rearrange Box(11,) space the last component is the
        # end-episode signal: any positive value terminates the episode
        # immediately with bad_called_terminate=True, which collapses
        # episode length to ~1 step for a randomly initialised policy and
        # makes baseline returns meaningless.  Baseline policies must not
        # control it, so it is pinned to "do not terminate".
        action = _suppress_terminate_action(action, self.env.action_space)
        result = self.env.step(action)
        if len(result) == 5:
            raw_obs, reward, terminated, truncated, info = result
        else:
            raw_obs, reward, done, info = result
            terminated = done
            truncated = False
        return self.observation(raw_obs), float(reward), bool(terminated), bool(truncated), info

    def close(self) -> None:
        """Close the underlying environment."""
        if hasattr(self.env, "close"):
            self.env.close()


# ---------------------------------------------------------------------------
# Vectorized env (mirrors the SB3 / CleanRL VecEnv interface minimal subset)
# ---------------------------------------------------------------------------
class HabitatVectorEnv:
    """Minimal vectorized environment wrapper around N Habitat envs.

    This is intentionally light-weight: it exposes ``reset``, ``step``,
    and ``close`` and returns stacked numpy arrays. It mirrors the
    subset of the Stable-Baselines3 ``VecEnv`` interface that the PPO
    trainer needs, without taking SB3 as a hard dependency.

    Each underlying environment is wrapped by
    :class:`VectorObservationWrapper`.
    """

    def __init__(
        self,
        env_fns: Sequence[Callable[[], gym.Env]],
        cfg: EnvConfig,
        mission_provider: Optional[MissionProvider] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """Construct a vectorized environment.

        Args:
            env_fns: Factory callables producing independent Habitat envs.
            cfg: Environment configuration.
            mission_provider: Shared mission provider injected into every
                sub-environment. The same mission embedding is broadcast
                to all agents in a sub-env (each sub-env caches per agent).
            device: Torch device for observation tensors.
        """
        if len(env_fns) == 0:
            raise ValueError("HabitatVectorEnv requires at least one env_fn")
        self.cfg = cfg
        self.num_envs = len(env_fns)
        # Social Rearrangement is a 2-agent task. A single mission embedding
        # is broadcast to both agents per sub-environment (each sub-env caches
        # an independent per-agent copy inside VectorObservationWrapper).
        self.num_agents = 2
        self.device = device or torch.device("cpu")
        self.envs: List[VectorObservationWrapper] = []
        for env_fn in env_fns:
            sub_env = env_fn()
            # ``make_env`` already returns a ``VectorObservationWrapper``.
            # Wrapping it a second time would feed the inner wrapper's output
            # (a list of tensors) back into ``observation()``, which expects
            # the raw Habitat dict -> all-zero observations. It would also
            # close the same underlying env twice.
            if isinstance(sub_env, VectorObservationWrapper):
                self.envs.append(sub_env)
            else:
                self.envs.append(
                    VectorObservationWrapper(
                        sub_env,
                        cfg=cfg,
                        mission_provider=mission_provider,
                        num_agents=2,
                        device=self.device,
                    )
                )
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int64)
        self.episode_success = np.zeros(self.num_envs, dtype=bool)

    # ------------------------------------------------------------------
    def reset(self) -> List[List[torch.Tensor]]:
        """Reset all sub-environments and return stacked observations."""
        obs_list: List[List[torch.Tensor]] = []
        for env in self.envs:
            obs, _ = env.reset()
            obs_list.append(obs if isinstance(obs, list) else [obs])
        self.episode_returns[:] = 0.0
        self.episode_lengths[:] = 0
        self.episode_success[:] = False
        return obs_list

    def step(
        self, actions: Sequence[Sequence[Any]]
    ) -> Tuple[List[List[torch.Tensor]], np.ndarray, np.ndarray,
               np.ndarray, Dict[str, Any]]:
        """Step all sub-environments with per-agent actions.

        Args:
            actions: ``actions[e][a]`` is the action for env ``e``,
                agent ``a``.

        Returns:
            Tuple ``(obs, rewards, dones, successes, info)`` where
            ``rewards`` and ``dones`` have shape ``(num_envs,)`` and
            represent the collective (team) reward and episode-termination
            flag respectively. ``obs[e]`` is the per-agent list of tensors.
        """
        next_obs: List[List[torch.Tensor]] = []
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        successes = np.zeros(self.num_envs, dtype=bool)
        infos: List[Dict[str, Any]] = []

        for i, env in enumerate(self.envs):
            obs, reward, terminated, truncated, info = env.step(actions[i])
            done = bool(terminated or truncated)
            success = bool(info.get("success", False)) if info else False

            self.episode_returns[i] += float(reward)
            self.episode_lengths[i] += 1
            self.episode_success[i] = self.episode_success[i] or success

            if done:
                info = dict(info or {})
                info["episode"] = {
                    "r": float(self.episode_returns[i]),
                    "l": int(self.episode_lengths[i]),
                    "s": bool(self.episode_success[i]),
                }
                self.episode_returns[i] = 0.0
                self.episode_lengths[i] = 0
                self.episode_success[i] = False
                obs, _ = env.reset()

            next_obs.append(obs if isinstance(obs, list) else [obs])
            rewards[i] = float(reward)
            dones[i] = done
            successes[i] = success
            infos.append(info)

        aggregate_info: Dict[str, Any] = {
            "infos": infos,
            "success_rate": float(successes.mean()),
        }
        return next_obs, rewards, dones, successes, aggregate_info

    def close(self) -> None:
        """Close all sub-environments."""
        for env in self.envs:
            try:
                env.close()
            except Exception:  # noqa: BLE001
                pass
        self.envs.clear()

    def __del__(self) -> None:  # pragma: no cover - best effort
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass
