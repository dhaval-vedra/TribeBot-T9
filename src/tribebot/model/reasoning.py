"""
Reasoning modules for TribeBot T9.
Combines the best ideas from T7 and T8 — all bugs fixed, networkx / scipy
dependencies removed (pure-PyTorch graph reasoning).

Modules
-------
RecursiveRefiner      — self-refinement with quality critique  (T7)
InternalDebate        — multi-agent internal consensus         (T7)
WorldModel            — predictive simulation                  (T7)
MetaCognition         — uncertainty-aware self-regulation      (T7)
GraphReasoner         — attention-based relational reasoning   (T8, no networkx)
CausalInference       — do-calculus-inspired causal reasoning  (T8, fixed)
MathematicalReasoner  — symbolic / proof reasoning             (T8, fixed)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# T7 — Recursive Self-Refinement
# ---------------------------------------------------------------------------

class RecursiveRefiner(nn.Module):
    """
    Iteratively refines hidden states using a critique score.
    Stops early when the critique exceeds a quality threshold.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.refine_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.critique_head = nn.Linear(embed_dim, 1)

    def forward(self, x: Tensor, max_iter: int = 3) -> Tensor:
        for _ in range(max_iter):
            quality = torch.sigmoid(self.critique_head(x.mean(dim=1))).mean()
            if quality.item() > 0.9:
                break
            delta = self.refine_net(x)
            x = x + delta * (1.0 - quality)
        return x


# ---------------------------------------------------------------------------
# T7 — Internal Multi-Agent Debate
# ---------------------------------------------------------------------------

class InternalDebate(nn.Module):
    """
    Runs num_agents independent views of the hidden state and synthesises
    a consensus via concatenation + linear projection.
    """

    def __init__(self, embed_dim: int, num_agents: int = 3) -> None:
        super().__init__()
        self.agents = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )
            for _ in range(num_agents)
        ])
        self.synthesiser = nn.Linear(embed_dim * num_agents, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        views = [agent(x) for agent in self.agents]
        return self.synthesiser(torch.cat(views, dim=-1))


# ---------------------------------------------------------------------------
# T7 — World Model (predictive simulation)
# ---------------------------------------------------------------------------

class WorldModel(nn.Module):
    """
    Rolls out a GRU-based predictive model from the last hidden state and
    blends valid simulations into the current representation.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.predictor = nn.GRU(embed_dim, embed_dim, num_layers=2, batch_first=True)
        self.validity_head = nn.Linear(embed_dim, 1)

    def forward(self, x: Tensor, steps: int = 3) -> tuple[Tensor, bool]:
        current = x[:, -1:, :]                   # [B, 1, C]
        hidden: Optional[Tensor] = None
        preds = []
        for _ in range(steps):
            out, hidden = self.predictor(current, hidden)
            preds.append(out)
            current = out
        sim = torch.cat(preds, dim=1)             # [B, steps, C]
        validity = torch.sigmoid(self.validity_head(sim.mean(dim=1))).mean()
        return sim, validity.item() > 0.65


# ---------------------------------------------------------------------------
# T7 — MetaCognition
# ---------------------------------------------------------------------------

class MetaCognition(nn.Module):
    """
    Estimates uncertainty and adjusts the hidden state amplitude accordingly.
    Tracks per-task goal progress in a simple float dictionary.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.self_model = nn.Linear(embed_dim, embed_dim)
        self.uncertainty_head = nn.Linear(embed_dim, 1)
        self.goal_tracker: Dict[str, float] = defaultdict(float)

    def forward(self, x: Tensor, task: str = "unknown") -> tuple[Tensor, float]:
        uncertainty = torch.sigmoid(self.uncertainty_head(x.mean(dim=1))).mean().item()
        if uncertainty > 0.6:
            x = self.self_model(x) * 1.25          # explore more
            self.goal_tracker[task] -= 0.05
        else:
            self.goal_tracker[task] += 0.03
        return x, uncertainty


# ---------------------------------------------------------------------------
# T8 — Graph Reasoner (pure-PyTorch, networkx removed)
# ---------------------------------------------------------------------------

class GraphReasoner(nn.Module):
    """
    Extracts a set of key concept vectors from the hidden sequence and
    performs attention-based message passing to build relational context.

    Replaces the networkx-based implementation from T8 with a fully
    differentiable, device-agnostic alternative.
    """

    def __init__(self, embed_dim: int, num_concepts: int = 16, num_mp_layers: int = 2) -> None:
        super().__init__()
        self.num_concepts = num_concepts

        # Concept selection: predict a score per token position
        self.concept_scorer = nn.Linear(embed_dim, 1)

        # Message-passing layers (self-attention among selected concepts)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=max(1, embed_dim // 64),
            dim_feedforward=embed_dim * 2,
            batch_first=True,
        )
        self.mp_layers = nn.TransformerEncoder(encoder_layer, num_layers=num_mp_layers)

        # Readout: collapse concept representations → one context vector
        self.readout = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        self.blend_weight = 0.1

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        k = min(self.num_concepts, T)

        # Select top-k concept tokens by L2 norm
        scores = self.concept_scorer(x).squeeze(-1)   # [B, T]
        _, top_idx = scores.topk(k, dim=-1)            # [B, k]

        # Gather concept embeddings  [B, k, C]
        idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, C)
        concepts = x.gather(1, idx_exp)

        # Message passing among concepts
        mp_out = self.mp_layers(concepts)              # [B, k, C]

        # Readout: mean, max, last
        ctx_mean = mp_out.mean(dim=1)                  # [B, C]
        ctx_max  = mp_out.max(dim=1).values            # [B, C]
        ctx_last = mp_out[:, -1, :]                    # [B, C]
        ctx = self.readout(torch.cat([ctx_mean, ctx_max, ctx_last], dim=-1))  # [B, C]

        # Broadcast back to sequence length and blend
        return x + ctx.unsqueeze(1).expand_as(x) * self.blend_weight


# ---------------------------------------------------------------------------
# T8 — Causal Inference (fixed)
# ---------------------------------------------------------------------------

class CausalInference(nn.Module):
    """
    Pearl do-calculus-inspired causal reasoning module.

    Bug fixed: original used `.diag()` on a [B, B] matrix which silently
    produced wrong shapes.  This version keeps causality per token using
    element-wise operations.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        # Learnable causal mixing matrix  [C, C]
        self.causal_kernel = nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.02)

        self.intervention_net = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.counterfactual_head = nn.Linear(embed_dim * 3, embed_dim)

    def _causal_strength(self, cause: Tensor, effect: Tensor) -> Tensor:
        """Scalar causal strength per batch item.  cause/effect: [B, C]"""
        transformed = torch.matmul(cause, self.causal_kernel)     # [B, C]
        # Element-wise product then mean → [B] scalar per sample
        strength = torch.sigmoid((transformed * effect).sum(dim=-1, keepdim=True))  # [B, 1]
        return strength

    def forward(self, x: Tensor, interventions: Optional[Tensor] = None) -> Tensor:
        B, T, C = x.shape

        causal_out = []
        for t in range(T):
            token = x[:, t, :]   # [B, C]

            if t > 0:
                strength = self._causal_strength(x[:, t - 1, :], token)  # [B, 1]
                token = token * strength                                    # broadcast fine

            if interventions is not None:
                iv = interventions[:, t, :]
                iv_active = iv.abs().sum(dim=-1, keepdim=True) > 0         # [B, 1]
                intervened = self.intervention_net(torch.cat([token, iv], dim=-1))
                token = torch.where(iv_active, intervened, token)

            causal_out.append(token)

        return torch.stack(causal_out, dim=1)   # [B, T, C]


# ---------------------------------------------------------------------------
# T8 — Mathematical Reasoner (fixed)
# ---------------------------------------------------------------------------

class MathematicalReasoner(nn.Module):
    """
    Lightweight neural module for mathematical / symbolic reasoning.

    Bug fixed: the original indexed `mathematical_content[b]` treating a
    Tensor like a Python list of per-batch items — replaced with proper
    tensor slicing.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.symbolic_encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.Tanh(),
        )
        self.equation_solver = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.proof_checker = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=max(1, embed_dim // 64),
                dim_feedforward=embed_dim * 2,
                batch_first=True,
            ),
            num_layers=2,
        )
        self.blend_weight = 0.1

    def forward(self, x: Tensor, math_context: Optional[Tensor] = None) -> Tensor:
        """
        x            : [B, T, C]  hidden states
        math_context : [B, T, C]  optional external math signal
        """
        B, T, C = x.shape
        source = math_context if math_context is not None else x

        # Encode and solve per batch item using tensor slicing (not list indexing)
        symbolic = self.symbolic_encoder(source)           # [B, T, 2C]
        solution = self.equation_solver(symbolic)          # [B, T, C]

        # Proof-check the solution sequence
        verified = self.proof_checker(solution)            # [B, T, C]

        return x + verified * self.blend_weight
