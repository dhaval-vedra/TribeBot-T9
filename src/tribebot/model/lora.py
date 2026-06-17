"""
LoRA (Low-Rank Adaptation) modules for TribeBot T9.
Includes single-rank LoRA and a multi-rank variant that blends
several ranks via learned attention weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LoRALayer(nn.Module):
    """Standard single-rank LoRA adapter: W + (B @ A) * scale."""

    def __init__(self, in_dim: int, out_dim: int, rank: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        self.rank = rank
        self.scale = alpha / rank

        self.lora_A = nn.Linear(in_dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_dim, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.lora_B(self.lora_A(x)) * self.scale


class MultiRankLoRA(nn.Module):
    """
    Blends multiple LoRA adapters of different ranks via soft attention.
    The model learns which rank granularity is most useful for each layer.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        ranks: list[int] | None = None,
        alpha: float = 16.0,
    ) -> None:
        super().__init__()
        if ranks is None:
            ranks = [4, 8, 16]
        self.ranks = ranks

        self.adapters = nn.ModuleList(
            [LoRALayer(in_dim, out_dim, rank=r, alpha=alpha) for r in ranks]
        )
        # Learned mixing weights (one scalar per rank)
        self.mix_logits = nn.Parameter(torch.zeros(len(ranks)))

    def forward(self, x: Tensor) -> Tensor:
        weights = F.softmax(self.mix_logits, dim=0)
        return sum(w * adapter(x) for w, adapter in zip(weights, self.adapters))
