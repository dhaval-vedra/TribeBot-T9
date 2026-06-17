"""
Attention mechanisms for TribeBot T9.

- MultiGateSwiGLU     : Enhanced feed-forward with multiple gating paths
- GroupedQueryAttention: GQA with optional Flash Attention v2 backend
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Optional Flash Attention — fall back to PyTorch SDPA if unavailable
try:
    from flash_attn import flash_attn_func as _flash_attn_func
    _FLASH_AVAILABLE = True
except ImportError:
    _FLASH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Feed-Forward with Multi-Gate SwiGLU
# ---------------------------------------------------------------------------

class MultiGateSwiGLU(nn.Module):
    """
    SwiGLU variant with num_gates extra gating paths whose outputs are
    averaged before modulating the main activation.  Gives the FFN richer
    gating dynamics without significantly increasing parameter count.
    """

    def __init__(self, dim: int, num_gates: int = 3) -> None:
        super().__init__()
        # Main gate: splits into (x, gate) pair
        self.main = nn.Linear(dim, dim * 2)
        # Extra learned gates applied to x
        self.extra_gates = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_gates)])

    def forward(self, x: Tensor) -> Tensor:
        h, gate = self.main(x).chunk(2, dim=-1)
        h = F.silu(h)
        gate = F.silu(gate)

        extra = torch.stack([F.silu(g(h)) for g in self.extra_gates], dim=-1).mean(dim=-1)
        return h * gate * extra


class FFNBlock(nn.Module):
    """Complete FFN sub-layer: project-up → MultiGateSwiGLU → project-down."""

    def __init__(self, embed_dim: int, ffn_mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        inner = embed_dim * ffn_mult
        self.up   = nn.Linear(embed_dim, inner)
        self.gate = MultiGateSwiGLU(inner)
        self.down = nn.Linear(inner, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.down(self.gate(F.gelu(self.up(x)))))


# ---------------------------------------------------------------------------
# Grouped Query Attention
# ---------------------------------------------------------------------------

class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (Ainslie et al., 2023) with:
      - Learnable RoPE (injected at call time via cos / sin tensors)
      - Multi-rank LoRA on Q, KV, and output projections
      - KV cache for autoregressive decoding
      - Flash Attention v2 backend when available, else PyTorch SDPA

    Parameters
    ----------
    embed_dim : model hidden dimension
    num_heads : number of query heads
    num_kv_groups : number of key-value head groups (must divide num_heads evenly)
    dropout : attention dropout probability (training only)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_groups: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = embed_dim // num_heads
        self.kv_head_dim = self.head_dim * num_kv_groups
        self.scale = self.head_dim ** -0.5
        self.dropout = dropout

        self.q_proj  = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj  = nn.Linear(embed_dim, self.kv_head_dim, bias=False)
        self.v_proj  = nn.Linear(embed_dim, self.kv_head_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.attn_drop = nn.Dropout(dropout)

        # Per-layer KV cache (filled during generate)
        self._k_cache: Optional[Tensor] = None
        self._v_cache: Optional[Tensor] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """x: [B, H, T, D].  cos/sin: [T, D]."""
        cos = cos.unsqueeze(0).unsqueeze(0)   # [1,1,T,D]
        sin = sin.unsqueeze(0).unsqueeze(0)
        d = x.shape[-1]
        x1, x2 = x[..., : d // 2], x[..., d // 2 :]
        rot = torch.cat([-x2, x1], dim=-1)
        return x * cos + rot * sin

    def reset_cache(self) -> None:
        self._k_cache = None
        self._v_cache = None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        use_cache: bool = False,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        B, T, C = x.shape

        q = self.q_proj(x)   # [B, T, C]
        k = self.k_proj(x)   # [B, T, kv_head_dim]
        v = self.v_proj(x)

        # Reshape to [B, H, T, head_dim]
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_kv_groups, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_kv_groups, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)

        # KV cache concatenation
        if use_cache:
            if self._k_cache is not None:
                k = torch.cat([self._k_cache, k], dim=2)
                v = torch.cat([self._v_cache, v], dim=2)
            self._k_cache = k.detach()
            self._v_cache = v.detach()

        # Expand KV groups to match query heads
        expand_factor = self.num_heads // self.num_kv_groups
        k = k.repeat_interleave(expand_factor, dim=1)   # [B, H, S, head_dim]
        v = v.repeat_interleave(expand_factor, dim=1)

        S = k.shape[2]   # key/value sequence length (may be > T when cache active)

        # ---- Compute attention ----
        if _FLASH_AVAILABLE and not use_cache and attn_mask is None:
            # Flash Attention expects [B, T, H, D] — transpose back
            q_fa = q.transpose(1, 2).contiguous()
            k_fa = k.transpose(1, 2).contiguous()
            v_fa = v.transpose(1, 2).contiguous()
            drop_p = self.dropout if self.training else 0.0
            attn_out = _flash_attn_func(q_fa, k_fa, v_fa, dropout_p=drop_p, causal=True)
            attn_out = attn_out.reshape(B, T, C)
        else:
            # PyTorch 2.0 Scaled Dot-Product Attention (fused, memory-efficient)
            is_causal = (attn_mask is None)
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal,
            )                                              # [B, H, T, head_dim]
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(attn_out)
