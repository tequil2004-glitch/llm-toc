"""Environment construction and vector-observation wrappers.

This package provides:
    * :class:`VectorObservationWrapper` - flattens and concatenates agent
      state, co-agent relative state, object/target relative positions,
      and the cached mission embedding vector into a single 1-D tensor.
    * :func:`make_env` - constructs a (vectorized) Habitat Social
      Rearrangement environment wrapped for MARL training.
"""

from llm_toc.env.habitat_wrapper import (
    HabitatVectorEnv,
    VectorObservationWrapper,
    VectorObs,
)
from llm_toc.env.env_factory import make_env, make_vec_env

__all__ = [
    "VectorObservationWrapper",
    "VectorObs",
    "HabitatVectorEnv",
    "make_env",
    "make_vec_env",
]
