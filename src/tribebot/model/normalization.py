"""
Normalization layers for TribeBot T9.
Includes standard RMSNorm and an advanced learnable variant.
"""

import torch
import torch.nn as nn
from torch import Tensor


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class AdvancedRMSNorm(nn.Module):
    """
    Enhanced RMSNorm with learnable per-element scale and global affine
    parameters (alpha / beta), giving the model extra expressive freedom
    at normalisation points without the mean-shift cost of LayerNorm.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor) -> Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        normalised = x * rms * self.weight
        return self.alpha * normalised + self.beta
