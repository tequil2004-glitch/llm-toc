"""Value surprise trigger based on the temporal-difference error.

At each step ``t`` the trigger computes

.. math::

    \\delta_t = r_t + \\gamma V(s_{t+1}; \\phi) - V(s_t; \\phi)

and compares :math:`|\\delta_t|` with a dynamic threshold
:math:`\\theta`. If :math:`|\\delta_t| > \\theta` the slow (LLM)
pipeline is triggered; otherwise the cached ``current_mission_buffer``
vector is reused.

The threshold is *dynamic*: it tracks the rolling mean and standard
deviation of :math:`|\\delta_t|` via exponential moving averages, so the
trigger adapts to drift in reward scale and value-function calibration.
This prevents both under-triggering (stale missions) and over-triggering
(expensive LLM thrash). A static floor ``trigger_min_threshold`` is
enforced to avoid firing on numerical noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from llm_toc.config import HierarchicalConfig


# ---------------------------------------------------------------------------
# Trigger decision
# ---------------------------------------------------------------------------
@dataclass
class TriggerDecision:
    """Outcome of a single trigger check.

    Attributes:
        triggered: True if the slow (LLM) pipeline should fire.
        td_error: The raw TD error :math:`\\delta_t`.
        abs_td_error: :math:`|\\delta_t|`.
        threshold: The dynamic threshold :math:`\\theta` used for this check.
        ema_mean: Current EMA of :math:`|\\delta_t|`.
        ema_std: Current EMA of the spread of :math:`|\\delta_t|`.
    """

    triggered: bool
    td_error: float
    abs_td_error: float
    threshold: float
    ema_mean: float
    ema_std: float


# ---------------------------------------------------------------------------
# Trigger
# ---------------------------------------------------------------------------
class TDErrorTrigger:
    """Dynamic TD-error surprise trigger.

    The trigger is stateful: it maintains exponential moving averages of
    the absolute TD error and its spread, and exposes a single method
    :meth:`check` that returns a :class:`TriggerDecision`.
    """

    def __init__(self, cfg: HierarchicalConfig) -> None:
        """Initialize the trigger.

        Args:
            cfg: Hierarchical-engine configuration.
        """
        self.cfg = cfg
        self.ema_mean: float = 0.0
        self.ema_var: float = 1.0
        self.initialized: bool = False
        # Number of observations fed to the EMA (for diagnostics).
        self.n_observations: int = 0

    # ------------------------------------------------------------------
    def _update_ema(self, abs_delta: float) -> None:
        """Update the EMA mean/variance of :math:`|\\delta_t|`."""
        alpha = self.cfg.trigger_ema_alpha
        if not self.initialized:
            self.ema_mean = abs_delta
            self.ema_var = 0.0
            self.initialized = True
        else:
            # EMA of the mean.
            self.ema_mean = (1.0 - alpha) * self.ema_mean + alpha * abs_delta
            # EMA of the variance (using the running mean).
            diff = abs_delta - self.ema_mean
            self.ema_var = (1.0 - alpha) * self.ema_var + alpha * (diff * diff)
        self.n_observations += 1

    @property
    def ema_std(self) -> float:
        """Return the current EMA standard deviation."""
        return float(max(self.ema_var, 0.0) ** 0.5)

    # ------------------------------------------------------------------
    def _dynamic_threshold(self) -> float:
        """Compute the dynamic threshold :math:`\\theta`.

        Two modes are supported:

        * **EMA-based** (default, ``trigger_use_ema_stats=True``):
            :math:`\\theta = \\mu_{\\text{EMA}} + k \\cdot \\sigma_{\\text{EMA}}`,
            where ``k = trigger_threshold_scale``.
        * **Static**: ``theta = trigger_threshold_init``.
        """
        if not self.cfg.trigger_use_ema_stats or not self.initialized:
            return max(self.cfg.trigger_threshold_init,
                       self.cfg.trigger_min_threshold)
        theta = self.ema_mean + self.cfg.trigger_threshold_scale * self.ema_std
        return max(theta, self.cfg.trigger_min_threshold)

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset all internal statistics."""
        self.ema_mean = 0.0
        self.ema_var = 1.0
        self.initialized = False
        self.n_observations = 0

    # ------------------------------------------------------------------
    def check(
        self,
        reward: float,
        value_t: float,
        value_tp1: float,
        gamma: float,
    ) -> TriggerDecision:
        """Check whether the slow system should trigger at step ``t``.

        Args:
            reward: Scalar reward :math:`r_t`.
            value_t: Value estimate :math:`V(s_t)`.
            value_tp1: Value estimate :math:`V(s_{t+1})`.
            gamma: Discount factor :math:`\\gamma`.

        Returns:
            A :class:`TriggerDecision` describing the outcome.
        """
        delta = float(reward) + float(gamma) * float(value_tp1) - float(value_t)
        abs_delta = abs(delta)
        self._update_ema(abs_delta)
        threshold = self._dynamic_threshold()
        triggered = abs_delta > threshold
        return TriggerDecision(
            triggered=triggered,
            td_error=delta,
            abs_td_error=abs_delta,
            threshold=threshold,
            ema_mean=self.ema_mean,
            ema_std=self.ema_std,
        )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def check_from_tensors(
        self,
        reward: torch.Tensor,
        value_t: torch.Tensor,
        value_tp1: torch.Tensor,
        gamma: float,
    ) -> TriggerDecision:
        """Tensor convenience overload that squeezes to Python scalars.

        Useful when values come straight from the critic network. The
        ``torch.no_grad`` decorator ensures this never builds a graph.
        """
        return self.check(
            reward=float(reward.detach().float().mean().item()),
            value_t=float(value_t.detach().float().mean().item()),
            value_tp1=float(value_tp1.detach().float().mean().item()),
            gamma=gamma,
        )
