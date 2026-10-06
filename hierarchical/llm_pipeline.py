"""Frozen LLM + frozen text-encoder pipeline for mission generation.

This module wraps:

1. A frozen, optionally LoRA-adapted open-source LLM
   (default: ``Qwen/Qwen2.5-7B-Instruct``) that consumes a MiniGrid-aligned
   textified prompt and produces a concise strategic mission string.
2. A frozen text encoder
   (default: ``sentence-transformers/all-MiniLM-L6-v2``, ``d=384``) that
   encodes the mission string into ``E_mission`` for concatenation into
   the policy observation vector.

Both models are *frozen*: their parameters never receive gradients. The
LLM is loaded in inference mode with ``torch.inference_mode`` guards and
the text encoder exposes only detached tensors.

A higher-level :class:`HierarchicalMissionManager` orchestrates the
trigger + saliency + textification + LLM pipeline and exposes a single
``get_mission`` callable that the environment wrapper uses as its
``mission_provider``. The manager caches the most recent mission
embedding and returns it on the fast path (no LLM call) when the TD-error
trigger does not fire.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

import torch
import torch.nn as nn

from llm_toc.config import EnvConfig, HierarchicalConfig
from llm_toc.hierarchical.saliency import JacobianSaliencyFilter, SaliencyResult
from llm_toc.hierarchical.textification import MiniGridTextifier
from llm_toc.hierarchical.trigger import TDErrorTrigger, TriggerDecision
from llm_toc.models.encoders import FrozenTextEncoder


# ---------------------------------------------------------------------------
# Frozen LLM wrapper
# ---------------------------------------------------------------------------
class FrozenLLMPipeline:
    """Wraps a frozen, optionally LoRA-adapted LLM for mission generation.

    The LLM is loaded lazily on first use so that importing this module
    does not require GPU/transformers. When ``transformers`` or ``peft``
    is unavailable, a deterministic stub generator is used (echoing the
    textified prompt) so the codebase remains importable on CI.

    Thread safety: a single :class:`threading.Lock` serializes LLM
    forward passes. Generation is intrinsically stateless given the
    prompt, but the underlying HF model is not always thread-safe.
    """

    def __init__(
        self,
        cfg: HierarchicalConfig,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize the pipeline (model loads lazily).

        Args:
            cfg: Hierarchical-engine configuration.
            device: Torch device for inference.
        """
        self.cfg = cfg
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = None
        self._model = None
        self._generation_lock = threading.Lock()
        self._fallback = False
        self._loaded = False

    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        """Lazily load the LLM and tokenizer (or fall back)."""
        if self._loaded:
            return
        self._loaded = True
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            dtype = self.cfg.llm_torch_dtype()
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.cfg.llm_model_id, trust_remote_code=True)
            if self._tokenizer.pad_token_id is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            self._model = AutoModelForCausalLM.from_pretrained(
                self.cfg.llm_model_id,
                torch_dtype=dtype,
                device_map=str(self.device),
                trust_remote_code=True,
            )
            self._model.eval()
            for p in self._model.parameters():
                p.requires_grad_(False)

            # Attach LoRA adapter if configured.
            if self.cfg.llm_lora_path:
                try:
                    from peft import PeftModel
                    self._model = PeftModel.from_pretrained(
                        self._model, self.cfg.llm_lora_path)
                    self._model.eval()
                except Exception:  # noqa: BLE001
                    # Continue without LoRA if peft is unavailable.
                    pass
        except Exception:  # noqa: BLE001 - fall back gracefully
            self._model = None
            self._tokenizer = None
            self._fallback = True

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def generate(self, prompt: str) -> str:
        """Generate a concise mission string from a MiniGrid-style prompt.

        Args:
            prompt: The textified situation prompt.

        Returns:
            A short strategic mission string (one sentence). When the LLM
            is unavailable, a deterministic stub is returned.
        """
        self._ensure_loaded()
        if self._fallback or self._model is None or self._tokenizer is None:
            return self._stub_generate(prompt)

        # Build a chat-style prompt (Qwen uses ChatML).
        messages = [
            {"role": "system",
             "content": ("You are a strategic planner for a cooperative "
                         "multi-agent rearrangement robot. Output one "
                         "concise mission sentence.")},
            {"role": "user", "content": prompt},
        ]
        try:
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except Exception:  # noqa: BLE001
            text = f"SYSTEM: Strategic planner.\nUSER: {prompt}\nASSISTANT:"

        with self._generation_lock:
            inputs = self._tokenizer(text, return_tensors="pt").to(
                self._model.device)
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.cfg.llm_max_new_tokens,
                do_sample=False,  # Greedy for reproducibility.
                pad_token_id=self._tokenizer.pad_token_id,
            )
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        mission = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return mission.strip() or self._stub_generate(prompt)

    # ------------------------------------------------------------------
    @staticmethod
    def _stub_generate(prompt: str) -> str:
        """Deterministic fallback mission (LLM unavailable)."""
        # Extract any "Object ID N" mentions and produce a routing mission.
        import re
        ids = re.findall(r"Object ID (\d+)", prompt)
        if ids:
            uniq = list(dict.fromkeys(ids))
            return (
                f"Prioritize placing object {uniq[0]} at its target zone "
                f"while coordinating with the co-agent."
            )
        return "Approach the nearest unplaced object and align with the co-agent."


# ---------------------------------------------------------------------------
# Hierarchical mission manager
# ---------------------------------------------------------------------------
@dataclass
class HierarchicalMetrics:
    """Lightweight counters for the hierarchical engine."""

    triggers_total: int = 0
    triggers_skipped: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    last_td_error: float = 0.0
    last_threshold: float = 0.0
    last_mission_string: str = ""


class HierarchicalMissionManager:
    """Orchestrates trigger + saliency + textification + LLM + encoder.

    The manager exposes a single :meth:`get_mission` method that returns
    the (possibly cached) mission embedding ``E_mission``. It also exposes
    :meth:`observe_transition`, which the trainer calls after every step
    with ``(reward, value_t, value_tp1, obs)``. The manager internally
    decides whether to invoke the slow LLM path or to return the cached
    embedding.

    The manager is the single ``mission_provider`` injected into the
    environment wrapper.

    Attributes:
        metrics: Live :class:`HierarchicalMetrics` for logging.
    """

    def __init__(
        self,
        hier_cfg: HierarchicalConfig,
        env_cfg: EnvConfig,
        critic: Optional[nn.Module] = None,
        text_encoder: Optional[FrozenTextEncoder] = None,
        llm: Optional[FrozenLLMPipeline] = None,
        device: Optional[torch.device] = None,
        gamma: float = 0.99,
    ) -> None:
        """Initialize the manager.

        Args:
            hier_cfg: Hierarchical-engine configuration.
            env_cfg: Environment configuration.
            critic: The (frozen-for-saliency) value network. Required for
                saliency computation; may be replaced at runtime via
                :meth:`attach_critic`.
            text_encoder: Frozen text encoder producing ``E_mission``.
            llm: Frozen LLM pipeline producing the mission string.
            device: Torch device.
            gamma: Discount factor used by the TD-error trigger.
        """
        self.hier_cfg = hier_cfg
        self.env_cfg = env_cfg
        self.device = device or torch.device("cpu")
        self.gamma = gamma
        self.metrics = HierarchicalMetrics()

        self.critic = critic
        self.text_encoder = text_encoder or FrozenTextEncoder(
            model_id=hier_cfg.text_encoder_id,
            device=self.device,
            expected_dim=hier_cfg.text_encoder_dim,
        )
        self.llm = llm or FrozenLLMPipeline(hier_cfg, device=self.device)
        self.trigger = TDErrorTrigger(hier_cfg)
        self.saliency = JacobianSaliencyFilter(hier_cfg, env_cfg)
        self.textifier = MiniGridTextifier(env_cfg)

        # Cached mission state.
        self._cached_mission_string: str = ""
        self._cached_mission_vector: Optional[torch.Tensor] = None
        self._cache_age_steps: int = 0
        self._last_flat_obs: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    def attach_critic(self, critic: nn.Module) -> None:
        """Replace the value network used for saliency.

        Useful when the trainer updates the critic weights: the manager
        always reads the *current* module, so this is only needed when
        the module instance itself changes.
        """
        self.critic = critic

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear caches and trigger statistics (call at episode reset)."""
        self.trigger.reset()
        self._cached_mission_string = ""
        self._cached_mission_vector = None
        self._cache_age_steps = 0
        self._last_flat_obs = None
        self.metrics = HierarchicalMetrics()

    # ------------------------------------------------------------------
    def observe_transition(
        self,
        reward: float,
        value_t: float,
        value_tp1: float,
        flat_obs: torch.Tensor,
        force_refresh: bool = False,
    ) -> TriggerDecision:
        """Feed a transition to the trigger and refresh mission if needed.

        This is the main entry point the trainer calls after every env
        step. When the trigger fires, the manager:

        1. Computes Jacobian saliency on ``flat_obs``.
        2. Textifies the result into a MiniGrid-style prompt.
        3. Calls the frozen LLM to produce a mission string.
        4. Encodes the string with the frozen text encoder.
        5. Caches the resulting ``E_mission``.

        On the fast path (no trigger), the cached embedding is reused
        and the LLM is not invoked.

        Args:
            reward: Scalar reward ``r_t``.
            value_t: ``V(s_t)``.
            value_tp1: ``V(s_{t+1})``.
            flat_obs: ``(obs_dim,)`` flat observation (with mission tail).
            force_refresh: If True, bypass the trigger and refresh now.

        Returns:
            The :class:`TriggerDecision` describing this step's outcome.
        """
        self._last_flat_obs = flat_obs.detach().to(self.device).float()
        decision = self.trigger.check(reward, value_t, value_tp1, self.gamma)
        self.metrics.last_td_error = decision.td_error
        self.metrics.last_threshold = decision.threshold
        self._cache_age_steps += 1

        stale = self._cache_age_steps > self.hier_cfg.mission_cache_max_age
        if decision.triggered or force_refresh or stale:
            self.metrics.triggers_total += 1
            self._refresh_mission(decision)
            self._cache_age_steps = 0
        else:
            self.metrics.triggers_skipped += 1
            if self._cached_mission_vector is not None:
                self.metrics.cache_hits += 1
            else:
                # Cold-start miss: generate a default mission.
                self.metrics.cache_misses += 1
                self._refresh_mission(decision)
                self._cache_age_steps = 0
        return decision

    # ------------------------------------------------------------------
    def _refresh_mission(self, decision: TriggerDecision) -> None:
        """Run the saliency -> textification -> LLM -> encoder pipeline."""
        if self.critic is None:
            mission_string = (
                "Default mission: approach the nearest unplaced object."
            )
        else:
            obs = self._last_flat_obs
            if obs is None:
                mission_string = (
                    "Default mission: approach the nearest unplaced object."
                )
            else:
                saliency = self.saliency.compute(self.critic, obs.unsqueeze(0))
                prompt = self.textifier.textify(
                    flat_obs=obs,
                    saliency=saliency,
                    extra_context=(
                        f"TD-error={decision.td_error:.3f}, "
                        f"threshold={decision.threshold:.3f}."
                    ),
                )
                mission_string = self.llm.generate(prompt)

        self._cached_mission_string = mission_string
        self.metrics.last_mission_string = mission_string
        self._cached_mission_vector = self.text_encoder.encode(
            mission_string).squeeze(0)

    # ------------------------------------------------------------------
    def get_mission(self) -> Optional[torch.Tensor]:
        """Return the current (possibly cached) mission embedding.

        This is the ``mission_provider`` callable injected into the
        environment wrapper. Returns ``None`` if no mission has been
        generated yet (the wrapper will then use its own zero-vector
        cold-start fallback).
        """
        if self._cached_mission_vector is None:
            return None
        return self._cached_mission_vector.detach().to(self.device).float()

    # ------------------------------------------------------------------
    def metrics_as_dict(self) -> dict:
        """Return metrics as a flat dict for logging."""
        m = self.metrics
        return {
            "hier/triggers_total": float(m.triggers_total),
            "hier/triggers_skipped": float(m.triggers_skipped),
            "hier/cache_hits": float(m.cache_hits),
            "hier/cache_misses": float(m.cache_misses),
            "hier/last_td_error": float(m.last_td_error),
            "hier/last_threshold": float(m.last_threshold),
        }
