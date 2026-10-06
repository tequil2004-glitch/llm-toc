"""Fast-slow hierarchical engine (the LLM-TOC core innovation).

The engine activates the slow (LLM) system only when the fast (RL)
system encounters *cognitive surprise* - quantified as a TD-error above
a dynamic threshold. When triggered, a Jacobian saliency filter locates
the entity in the vector observation responsible for the surprise, the
result is textified into a MiniGrid-aligned prompt, and a frozen LLM
produces a fresh strategic mission string that is re-encoded into the
mission embedding ``E_mission``.
"""

from llm_toc.hierarchical.trigger import TDErrorTrigger, TriggerDecision
from llm_toc.hierarchical.saliency import JacobianSaliencyFilter
from llm_toc.hierarchical.textification import MiniGridTextifier
from llm_toc.hierarchical.llm_pipeline import (
    FrozenLLMPipeline,
    HierarchicalMissionManager,
)

__all__ = [
    "TDErrorTrigger",
    "TriggerDecision",
    "JacobianSaliencyFilter",
    "MiniGridTextifier",
    "FrozenLLMPipeline",
    "HierarchicalMissionManager",
]
