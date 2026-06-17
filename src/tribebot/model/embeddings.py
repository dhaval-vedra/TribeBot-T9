"""
Embedding modules for TribeBot T9.
- LearnableRoPE  : Rotary Position Embedding with learnable frequency bias
- SemanticDynamicVocab : Token embedding that supports runtime vocabulary expansion
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Rotary Position Embedding
# ---------------------------------------------------------------------------

class LearnableRoPE(nn.Module):
    """
    Rotary Position Embedding (Su et al., 2021) with a small learnable
    frequency perturbation so the model can adapt its positional bias
    during training.
    """

    def __init__(self, head_dim: int, max_seq_len: int = 262144, base: float = 10000.0) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        # Fixed inverse-frequency buffer (not a parameter)
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)

        # Small learnable bias on top of the fixed frequencies
        self.freq_bias = nn.Parameter(torch.zeros(head_dim // 2) * 0.01)

        self._cos_cache: Optional[Tensor] = None
        self._sin_cache: Optional[Tensor] = None
        self._cached_len: int = 0

    def _build_cache(self, seq_len: int, device: torch.device) -> None:
        if seq_len <= self._cached_len:
            return
        inv_freq = self.inv_freq + 0.05 * torch.tanh(self.freq_bias)
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        angles = torch.outer(positions, inv_freq)          # [T, D/2]
        emb = torch.cat([angles, angles], dim=-1)           # [T, D]
        self._cos_cache = emb.cos()
        self._sin_cache = emb.sin()
        self._cached_len = seq_len

    def forward(self, seq_len: int, device: torch.device) -> Tuple[Tensor, Tensor]:
        self._build_cache(seq_len, device)
        return self._cos_cache[:seq_len], self._sin_cache[:seq_len]  # type: ignore[index]

    @staticmethod
    def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """Apply RoPE to query / key tensors.  x: [B, H, T, D]"""
        d = x.shape[-1]
        x1 = x[..., : d // 2]
        x2 = x[..., d // 2 :]
        # cos/sin: [T, D] → broadcast over B, H
        cos = cos.unsqueeze(0).unsqueeze(0)   # [1, 1, T, D]
        sin = sin.unsqueeze(0).unsqueeze(0)
        rotated = torch.cat([-x2, x1], dim=-1)
        return x * cos + rotated * sin


# ---------------------------------------------------------------------------
# Dynamic Vocabulary Embedding
# ---------------------------------------------------------------------------

class SemanticDynamicVocab(nn.Module):
    """
    Token embedding that supports runtime expansion of the vocabulary.
    New tokens can be added with optional semantic initialisation from
    an existing embedding vector.
    """

    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.base_vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

        # Extra embeddings added at runtime (not registered as parameters so
        # they survive module.to(device) calls without confusion)
        self._extra_weight: Optional[Tensor] = None
        self._token_map: dict[str, int] = {}
        self._next_id: int = vocab_size

    @property
    def vocab_size(self) -> int:
        base = self.embedding.num_embeddings
        extra = 0 if self._extra_weight is None else self._extra_weight.shape[0]
        return base + extra

    def add_token(self, token: str, init_vector: Optional[Tensor] = None) -> int:
        """Register a new token.  Returns the assigned token id."""
        if token in self._token_map:
            return self._token_map[token]

        if init_vector is None:
            new_embed = torch.randn(1, self.embed_dim, device=self.embedding.weight.device) * 0.02
        else:
            new_embed = init_vector.detach().view(1, self.embed_dim).to(self.embedding.weight.device)

        if self._extra_weight is None:
            self._extra_weight = new_embed
        else:
            self._extra_weight = torch.cat([self._extra_weight, new_embed], dim=0)

        token_id = self._next_id
        self._token_map[token] = token_id
        self._next_id += 1
        return token_id

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, T] integer token ids."""
        base_size = self.embedding.num_embeddings

        # Ids within the base vocabulary → standard embedding lookup
        base_mask = x < base_size
        out = torch.zeros(*x.shape, self.embed_dim, device=x.device, dtype=self.embedding.weight.dtype)
        if base_mask.any():
            out[base_mask] = self.embedding(x[base_mask].clamp(0, base_size - 1))

        # Ids in the extended vocabulary
        extra_mask = ~base_mask
        if extra_mask.any() and self._extra_weight is not None:
            extra_ids = (x[extra_mask] - base_size).clamp(0, self._extra_weight.shape[0] - 1)
            out[extra_mask] = self._extra_weight[extra_ids]

        return out
