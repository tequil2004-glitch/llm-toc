"""MiniGrid-aligned textification of saliency results.

Translates the high-gradient entity indices produced by
:class:`JacobianSaliencyFilter` together with their relative spatial
coordinates into a structured natural-language prompt. The output
mirrors the MiniGrid alignment syntax used by language-conditioned
MARL benchmarks so that the downstream frozen LLM receives a
well-formed, tokenizable situation description.

Example output::

    "The agent is at coordinate (1.20, 0.50, 0.00) facing (0.0, 0.0, 0.0, 1.0).
     The co-agent is 1.50 steps ahead and 0.30 steps to the left.
     The critical object ID 0 is 1.20 steps ahead and 0.10 steps to the right,
     its target is 2.50 steps behind. Object ID 3 is 0.80 steps to the left."

The textifier is deterministic and side-effect-free; it can be unit-tested
without GPU or LLM dependencies.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch

from llm_toc.config import EnvConfig
from llm_toc.env.habitat_wrapper import VectorObs
from llm_toc.hierarchical.saliency import SaliencyResult


# ---------------------------------------------------------------------------
# Spatial language helpers
# ---------------------------------------------------------------------------
def _quaternion_to_heading_deg(quat: Sequence[float]) -> float:
    """Convert a ``(w, x, y, z)`` quaternion to a heading angle in degrees.

    Uses the yaw component (rotation about the world up axis). Returns
    ``0.0`` for degenerate inputs.
    """
    w, x, y, z = (float(v) for v in quat)
    # yaw = atan2(2(w z + x y), 1 - 2(y^2 + z^2))
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    if abs(cosy_cosp) < 1e-12 and abs(siny_cosp) < 1e-12:
        return 0.0
    import math
    return float(math.degrees(math.atan2(siny_cosp, cosy_cosp)))


def _relative_direction_words(
    rel_pos: Sequence[float], heading_deg: float = 0.0
) -> str:
    """Translate a 3-D relative position into MiniGrid-style prose.

    The relative vector is rotated into the agent's local frame using
    ``heading_deg`` (so "ahead"/"behind"/"left"/"right" are agent-centric).
    The vertical (z) component is reported as "above"/"below".

    Args:
        rel_pos: ``(dx, dy, dz)`` relative position in world frame.
        heading_deg: Agent heading in degrees (yaw about up axis).

    Returns:
        A prose description like ``"1.20 steps ahead and 0.30 steps to the
        left"``.
    """
    import math
    dx, dy, dz = (float(v) for v in rel_pos)
    theta = math.radians(heading_deg)
    # Rotate (dx, dy) by -theta to get agent-frame (forward, right).
    forward = dx * math.cos(theta) + dy * math.sin(theta)
    right = -dx * math.sin(theta) + dy * math.cos(theta)
    parts: List[str] = []
    if abs(forward) > 1e-3:
        direction = "ahead" if forward >= 0 else "behind"
        parts.append(f"{abs(forward):.2f} steps {direction}")
    if abs(right) > 1e-3:
        direction = "to the right" if right >= 0 else "to the left"
        parts.append(f"{abs(right):.2f} steps {direction}")
    if abs(dz) > 1e-3:
        direction = "above" if dz >= 0 else "below"
        parts.append(f"{abs(dz):.2f} steps {direction}")
    if not parts:
        return "at the agent's position"
    return " and ".join(parts)


# ---------------------------------------------------------------------------
# Textifier
# ---------------------------------------------------------------------------
class MiniGridTextifier:
    """Convert saliency results + observations into MiniGrid-style prompts.

    The textifier:

    1. Deconstructs the flat observation into structured fields using
       :class:`VectorObs`.
    2. Identifies which saliency indices correspond to which entities.
    3. Emits a single prompt string containing agent position, co-agent
       relative direction, and the relative direction of each salient
       object (and its target).
    """

    def __init__(self, env_cfg: EnvConfig) -> None:
        """Initialize the textifier.

        Args:
            env_cfg: Environment configuration (for entity widths).
        """
        self.env_cfg = env_cfg

    # ------------------------------------------------------------------
    def _format_agent(self, vobs: VectorObs) -> str:
        """Format the agent-state line of the prompt."""
        pos = vobs.agent_state[:3].tolist()
        quat = vobs.agent_state[3:7].tolist()
        return (
            f"The agent is at coordinate "
            f"({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) "
            f"facing ({quat[0]:.2f}, {quat[1]:.2f}, "
            f"{quat[2]:.2f}, {quat[3]:.2f})."
        )

    def _format_coagent(self, vobs: VectorObs, heading_deg: float) -> str:
        """Format the co-agent relative-direction line."""
        rel = vobs.coagent_rel[:3].tolist()
        words = _relative_direction_words(rel, heading_deg=heading_deg)
        return f"The co-agent is {words}."

    def _format_object(
        self,
        vobs: VectorObs,
        obj_id: int,
        heading_deg: float,
    ) -> Optional[str]:
        """Format a single salient object line. Returns None if invalid slot."""
        if obj_id >= self.env_cfg.max_objects:
            return None
        if not bool(vobs.valid_object_mask[obj_id].item()):
            return None
        row = vobs.objects[obj_id]
        rel_obj = row[:3].tolist()
        rel_tgt = row[3:6].tolist()
        obj_words = _relative_direction_words(rel_obj, heading_deg=heading_deg)
        tgt_words = _relative_direction_words(rel_tgt, heading_deg=heading_deg)
        return (
            f"Object ID {obj_id} is {obj_words}, "
            f"its target is {tgt_words}."
        )

    # ------------------------------------------------------------------
    def textify(
        self,
        flat_obs: torch.Tensor,
        saliency: SaliencyResult,
        extra_context: Optional[str] = None,
    ) -> str:
        """Build a MiniGrid-aligned prompt from an observation + saliency.

        Args:
            flat_obs: ``(obs_dim,)`` flat observation tensor (must include
                the mission embedding at the tail; it is ignored for
                textification).
            saliency: The :class:`SaliencyResult` produced by
                :class:`JacobianSaliencyFilter`.
            extra_context: Optional free-text context appended at the end
                (e.g., a recent mission statement).

        Returns:
            A single prompt string suitable for the frozen LLM.
        """
        vobs = VectorObs.from_flat(flat_obs, self.env_cfg)
        heading = _quaternion_to_heading_deg(vobs.agent_state[3:7].tolist())

        lines: List[str] = [self._format_agent(vobs)]
        lines.append(self._format_coagent(vobs, heading_deg=heading))

        # Collect unique object IDs from the saliency breakdown (preserve
        # order by magnitude, which is already descending).
        seen_ids: set[int] = set()
        salient_obj_ids: List[int] = []
        for kind, eid, _ in saliency.entity_breakdown:
            if kind == "object" and eid not in seen_ids:
                seen_ids.add(eid)
                salient_obj_ids.append(eid)

        if not salient_obj_ids:
            # No salient objects: include the first valid object as a fallback.
            for i in range(self.env_cfg.max_objects):
                if bool(vobs.valid_object_mask[i].item()):
                    salient_obj_ids.append(i)
                    break

        for oid in salient_obj_ids:
            line = self._format_object(vobs, oid, heading_deg=heading)
            if line is not None:
                lines.append(line)

        # Critical-object annotation (top-1 entity) for LLM prioritization.
        if saliency.entity_breakdown:
            top_kind, top_eid, _ = saliency.entity_breakdown[0]
            if top_kind == "object":
                lines.append(
                    f"The critical object ID {top_eid} requires immediate "
                    f"strategic attention."
                )
            elif top_kind == "coagent":
                lines.append(
                    "The co-agent's relative pose is the critical surprise "
                    "factor; coordinate strategically."
                )
            elif top_kind == "agent":
                lines.append(
                    "The agent's own proprioceptive state is the critical "
                    "surprise factor; reconsider the current maneuver."
                )

        if extra_context:
            lines.append(f"Context: {extra_context}")

        lines.append(
            "Propose a concise strategic mission for the agent's next "
            "sub-goal in one sentence."
        )
        return " ".join(lines)
