"""Jacobian-based input saliency filter.

When the TD-error trigger fires, the saliency filter computes the
gradient of the value-network output with respect to the *vector*
observation input:

.. math::

    \\text{Saliency} = \\left| \\frac{\\partial V(s)}{\\partial
    s_{\\text{vector}}} \\right|

The top-``k`` indices by absolute gradient magnitude identify which
entity/object in the vector observation is causing the cognitive
surprise. The trainer then maps those indices back to structured
entities (agent, co-agent, object slots) for textification.

CRITICAL MEMORY SAFETY
-----------------------
All gradient computations use ``create_graph=False`` and
``retain_graph=False`` so that the temporary computational graph built
for the saliency backward pass is freed immediately. This is essential
for long rollouts: failing to release the graph would leak memory and
eventually OOM on multi-hour training runs.

Additionally, the filter:

* Wraps the backward call in ``torch.autograd.grad`` (not
  ``tensor.backward``) for explicit control over inputs and outputs.
* Detaches the returned indices/magnitudes from the autograd graph.
* Re-enables ``torch.no_grad`` for the surrounding context to prevent
  accidental graph construction during indexing.
* Restores the ``requires_grad`` flag of the input to its prior state so
  the saliency computation is invisible to the rest of the training loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn

from llm_toc.config import EnvConfig, HierarchicalConfig


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
@dataclass
class SaliencyResult:
    """Output of a single saliency computation.

    Attributes:
        top_indices: Flat observation indices with the highest saliency,
            sorted descending by magnitude. Length ``<= top_k``.
        top_magnitudes: Saliency magnitudes aligned with ``top_indices``.
        full_saliency: The full ``(obs_dim,)`` saliency vector (detached).
            Useful for visualization / debugging.
        entity_breakdown: A list of ``(entity_kind, entity_id, sub_index)``
            tuples identifying which structured entity each top index
            belongs to. ``entity_kind`` is one of ``"agent"``,
            ``"coagent"``, ``"object"``, or ``"mission"``.
    """

    top_indices: torch.Tensor
    top_magnitudes: torch.Tensor
    full_saliency: torch.Tensor
    entity_breakdown: List[Tuple[str, int, int]]


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------
class JacobianSaliencyFilter:
    """Compute input-gradients of the value network and rank entities.

    The filter is constructed once and called with the current value
    network and a flat observation batch. It is stateless aside from its
    configuration.
    """

    def __init__(
        self,
        cfg: HierarchicalConfig,
        env_cfg: EnvConfig,
    ) -> None:
        """Initialize the filter.

        Args:
            cfg: Hierarchical-engine configuration (top_k, eps, etc.).
            env_cfg: Environment configuration (for entity index mapping).
        """
        self.cfg = cfg
        self.env_cfg = env_cfg
        self.top_k = cfg.top_k_salient

    # ------------------------------------------------------------------
    def _flat_index_to_entity(
        self, idx: int
    ) -> Tuple[str, int, int]:
        """Map a flat observation index to ``(entity_kind, entity_id, sub)``.

        Layout (mirrors :class:`VectorObservationWrapper`):

        * ``[0 : agent_state_dim)`` -> ``("agent", 0, sub)``
        * ``[agent_state_dim : agent_state_dim + coagent_rel_dim)`` ->
          ``("coagent", 0, sub)``
        * Next ``max_objects * object_dim`` slots -> ``("object", k, sub)``
          where ``k = (idx - offset) // object_dim`` and
          ``sub = (idx - offset) % object_dim``.
        * Last ``mission_embed_dim`` slots -> ``("mission", 0, sub)``.
        """
        c = self.env_cfg
        offset = 0
        if idx < c.agent_state_dim:
            return ("agent", 0, idx)
        offset += c.agent_state_dim
        if idx < offset + c.coagent_rel_dim:
            return ("coagent", 0, idx - offset)
        offset += c.coagent_rel_dim
        obj_region = c.max_objects * c.object_dim
        if idx < offset + obj_region:
            local = idx - offset
            return ("object", local // c.object_dim, local % c.object_dim)
        offset += obj_region
        if idx < offset + c.mission_embed_dim:
            return ("mission", 0, idx - offset)
        return ("unknown", -1, -1)

    # ------------------------------------------------------------------
    def compute(
        self,
        critic: nn.Module,
        obs: torch.Tensor,
    ) -> SaliencyResult:
        """Compute saliency for a single observation.

        Args:
            critic: The value network ``V(s)``. Must accept ``obs`` of
                shape ``(1, obs_dim)`` and return ``(1,)``.
            obs: ``(1, obs_dim)`` observation tensor.

        Returns:
            A :class:`SaliencyResult` with top-k indices and the entity
            breakdown.

        CRITICAL MEMORY SAFETY:
            * The input tensor's ``requires_grad`` is enabled only for the
              duration of this call and restored afterwards.
            * :func:`torch.autograd.grad` is called with
              ``create_graph=False`` and ``retain_graph=False`` so the
              temporary graph is freed immediately.
            * All returned tensors are detached from the autograd graph.
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if obs.shape[0] != 1:
            raise ValueError(
                f"Saliency expects batch size 1, got {obs.shape[0]}."
            )

        # Work on a fresh leaf tensor so we don't perturb the caller's
        # storage or requires_grad state.
        obs_leaf = obs.detach().clone().to(obs.device)
        prev_requires_grad = obs_leaf.requires_grad
        obs_leaf.requires_grad_(True)

        # CRITICAL: create_graph=False, retain_graph=False to prevent the
        # temporary backward graph from leaking into long rollouts.
        grad = torch.autograd.grad(
            outputs=critic(obs_leaf),
            inputs=obs_leaf,
            grad_outputs=torch.ones_like(critic(obs_leaf.detach())),
            create_graph=False,
            retain_graph=False,
            allow_unused=False,
        )[0]

        saliency = grad.detach().abs()
        # Normalize for numerical stability (eps added in denominator).
        saliency = saliency / (saliency.max() + self.cfg.saliency_eps)

        # Restore the leaf's requires_grad state.
        obs_leaf.requires_grad_(prev_requires_grad)

        k = min(self.top_k, int(saliency.shape[-1]))
        top_magnitudes, top_indices = torch.topk(saliency, k=k, dim=-1)
        top_indices = top_indices.detach().squeeze(0)
        top_magnitudes = top_magnitudes.detach().squeeze(0)
        saliency_flat = saliency.detach().squeeze(0)

        entity_breakdown: List[Tuple[str, int, int]] = [
            self._flat_index_to_entity(int(idx.item()))
            for idx in top_indices
        ]

        return SaliencyResult(
            top_indices=top_indices,
            top_magnitudes=top_magnitudes,
            full_saliency=saliency_flat,
            entity_breakdown=entity_breakdown,
        )
