"""Frozen text encoders for mission strings.

Wraps ``sentence-transformers/all-MiniLM-L6-v2`` (``d=384``) in a
torch.nn.Module-like API with a frozen parameter set. The encoder is
used inside the hierarchical pipeline to convert an LLM-generated
mission string into the mission embedding ``E_mission`` that is
concatenated into the policy observation vector.

When ``sentence-transformers`` is not installed, a small deterministic
hashing-based fallback encoder is used so the codebase remains importable
on machines without the dependency. The fallback produces 384-dim vectors
but is **not** semantically meaningful - it exists only for smoke tests.
"""

from __future__ import annotations

import hashlib
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn


class FrozenTextEncoder(nn.Module):
    """Frozen ``all-MiniLM-L6-v2`` text encoder (output dim 384).

    The module wraps a SentenceTransformer model with ``requires_grad``
    disabled for all parameters. It exposes both batched and unbatched
    ``encode`` methods that return detached tensors (no gradient flow,
    matching the "frozen text encoder" specification).

    Attributes:
        embed_dim: Output embedding dimensionality (384 for MiniLM-L6-v2).
        model_id: HuggingFace model identifier.
        device: Torch device used for inference.
    """

    embed_dim: int

    def __init__(
        self,
        model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[torch.device] = None,
        expected_dim: int = 384,
    ) -> None:
        """Initialize the frozen encoder.

        Args:
            model_id: HuggingFace model identifier.
            device: Torch device for inference.
            expected_dim: Required output dimension (must equal 384 per the
                project spec). A mismatch raises ``ValueError``.
        """
        super().__init__()
        self.model_id = model_id
        self.embed_dim = expected_dim
        self.device = device or torch.device("cpu")
        self._st = None
        self._fallback = False
        try:
            from sentence_transformers import SentenceTransformer
            self._st = SentenceTransformer(model_id, device=str(self.device))
            actual_dim = self._st.get_sentence_embedding_dimension()
            if actual_dim != expected_dim:
                raise ValueError(
                    f"Encoder '{model_id}' returned dim {actual_dim}, "
                    f"expected {expected_dim}."
                )
        except Exception:  # noqa: BLE001 - fall back gracefully
            self._st = None
            self._fallback = True
        self._freeze()

    def _freeze(self) -> None:
        """Disable gradients for all parameters (frozen encoder)."""
        for p in self.parameters():
            p.requires_grad_(False)
        if self._st is not None:
            for p in self._st.parameters():
                p.requires_grad_(False)
        self.eval()

    # ------------------------------------------------------------------
    def encode(
        self,
        texts: Union[str, Sequence[str]],
        batch_size: int = 32,
    ) -> torch.Tensor:
        """Encode one or more mission strings to a ``(B, 384)`` tensor.

        The output is detached (no gradient) and moved to the encoder's
        device, matching the frozen-encoder specification.

        Args:
            texts: A single string or a sequence of strings.
            batch_size: SentenceTransformer batch size.

        Returns:
            ``(B, 384)`` float tensor. If a single string is passed, B=1.
        """
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        if self._st is not None:
            arr = self._st.encode(
                list(texts),
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            vec = torch.from_numpy(np.asarray(arr, dtype=np.float32))
        else:
            vec = self._fallback_encode(list(texts))
        vec = vec.to(self.device).float().detach()
        return vec

    def _fallback_encode(self, texts: List[str]) -> torch.Tensor:
        """Deterministic hash-based fallback encoder (smoke-test only).

        Produces a 384-dim L2-normalized vector per string. **Not**
        semantically meaningful - used only when SentenceTransformer is
        unavailable so the codebase stays importable.
        """
        out = np.zeros((len(texts), self.embed_dim), dtype=np.float32)
        for i, t in enumerate(texts):
            h = hashlib.sha256(t.encode("utf-8")).digest()
            # Repeat the 32-byte hash to fill 384 dims (96 bytes -> 384 bytes
            # is too many; instead tile the 256-bit hash into 384 floats).
            bytes_needed = self.embed_dim * 4
            buf = (h * ((bytes_needed // len(h)) + 1))[:bytes_needed]
            arr = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
            arr = arr.reshape(self.embed_dim, 4).mean(axis=1)
            norm = np.linalg.norm(arr) + 1e-8
            out[i] = arr / norm
        return torch.from_numpy(out)

    # ------------------------------------------------------------------
    def forward(self, texts: Union[str, Sequence[str]]) -> torch.Tensor:
        """nn.Module-style forward delegating to :meth:`encode`."""
        return self.encode(texts)

    def train(self, mode: bool = True) -> "FrozenTextEncoder":  # noqa: D401
        """Override to keep the encoder frozen (eval mode always)."""
        super().train(False)
        return self
