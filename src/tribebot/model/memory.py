"""
Hierarchical Memory modules for TribeBot T9.
Merges the lightweight T7 memory with the multi-scale T8 memory into a
unified AdvancedHierarchicalMemory that operates at three temporal scales.
"""

from __future__ import annotations

from collections import deque
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Simple compressed chunk memory (T7 concept, fixed)
# ---------------------------------------------------------------------------

class ChunkCompressor(nn.Module):
    """Compresses a variable-length sequence chunk to a single vector."""

    def __init__(self, embed_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.act = nn.GELU()

    def forward(self, chunk: Tensor) -> Tensor:
        """chunk: [T, C]  →  [1, C]"""
        # Mean-pool then project
        pooled = chunk.mean(dim=0, keepdim=True)   # [1, C]
        return self.norm(self.act(self.linear(pooled)))


# ---------------------------------------------------------------------------
# Advanced three-scale memory (T8 concept, fully corrected)
# ---------------------------------------------------------------------------

class AdvancedHierarchicalMemory(nn.Module):
    """
    Three-scale episodic memory:
      - Short-term  : raw experiences (fast, small buffer)
      - Medium-term : transformer-encoded experiences
      - Long-term   : GRU-consolidated summaries

    Memory is retrieved via cross-attention between the current query and
    the combined memory pool.
    """

    def __init__(
        self,
        embed_dim: int,
        chunk_size: int = 512,
        st_capacity: int = 50,
        mt_capacity: int = 200,
        lt_capacity: int = 500,
        num_encoder_layers: int = 2,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.chunk_size = chunk_size

        # Buffers (store 1-D vectors, each shape [C])
        self.short_term: deque[Tensor] = deque(maxlen=st_capacity)
        self.medium_term: deque[Tensor] = deque(maxlen=mt_capacity)
        self.long_term: deque[Tensor] = deque(maxlen=lt_capacity)

        # Processing networks
        self.mt_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=max(1, embed_dim // 64),
                dim_feedforward=embed_dim * 2, batch_first=True
            ),
            num_layers=num_encoder_layers,
        )
        self.lt_consolidator = nn.GRU(embed_dim, embed_dim, batch_first=True)

        # Cross-attention retrieval: query=current hidden, key/value=memories
        self.retrieval_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=max(1, embed_dim // 64),
            batch_first=True,
        )

        # Gate controlling how much retrieved memory blends into the stream
        self.memory_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid(),
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_experience(self, x: Tensor) -> None:
        """x: [B, T, C] — process each batch item and store in all scales."""
        B, T, C = x.shape
        for b in range(B):
            seq = x[b].detach()                          # [T, C]
            summary = seq.mean(dim=0)                    # [C]

            # Short-term: raw mean vector
            self.short_term.append(summary)

            # Medium-term: transformer-encoded mean
            encoded = self.mt_encoder(seq.unsqueeze(0))  # [1, T, C]
            self.medium_term.append(encoded.squeeze(0).mean(dim=0))

            # Long-term: GRU consolidation over the last 10 short-term items
            if len(self.short_term) >= 10:
                recent = torch.stack(list(self.short_term)[-10:]).unsqueeze(0)  # [1, 10, C]
                _, hn = self.lt_consolidator(recent.to(x.device))
                self.long_term.append(hn.squeeze(0).squeeze(0))

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def retrieve(self, query: Tensor) -> Tensor:
        """
        query: [B, T, C]
        Returns: [B, T, C] memory context to be added to the hidden stream.
        """
        memories: List[Tensor] = []
        for pool in (self.short_term, self.medium_term, self.long_term):
            if len(pool) > 0:
                memories.append(torch.stack(list(pool)).to(query.device))

        if not memories:
            return torch.zeros_like(query)

        # Keys/values: [1, M, C]  (broadcast over batch)
        mem_kv = torch.cat(memories, dim=0).unsqueeze(0)   # [1, M, C]
        B, T, C = query.shape
        mem_kv = mem_kv.expand(B, -1, -1)                  # [B, M, C]

        # Query: use mean over T so each batch item queries with one vector
        q = query.mean(dim=1, keepdim=True)                 # [B, 1, C]
        retrieved, _ = self.retrieval_attn(q, mem_kv, mem_kv)  # [B, 1, C]
        retrieved = retrieved.expand(-1, T, -1)             # [B, T, C]

        # Gated blend
        gate = self.memory_gate(torch.cat([query, retrieved], dim=-1))  # [B, T, C]
        return gate * retrieved
