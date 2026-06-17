"""
Text generation utilities for TribeBot T9.
Provides sampling helpers, beam search scaffold, and batch generation.
"""

from __future__ import annotations

from typing import Any, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def top_k_filter(logits: Tensor, top_k: int) -> Tensor:
    """Zero out logits below the top-k threshold."""
    if top_k <= 0:
        return logits
    k = min(top_k, logits.size(-1))
    threshold = torch.topk(logits, k).values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def top_p_filter(logits: Tensor, top_p: float) -> Tensor:
    """Nucleus (top-p) filtering."""
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cum_prob > top_p (keep the first one over threshold)
    remove = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    # Scatter back
    return logits.scatter(-1, sorted_idx, sorted_logits)


def apply_repetition_penalty(logits: Tensor, generated_ids: Tensor, penalty: float) -> Tensor:
    """Divide logit of already-generated tokens by penalty (> 1 discourages repeats)."""
    if penalty == 1.0:
        return logits
    for token_id in set(generated_ids.tolist()):
        logits[..., token_id] /= penalty
    return logits


@torch.no_grad()
def generate_text(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1,
    device: Optional[torch.device] = None,
    task: str = "generation",
) -> str:
    """
    Standalone generation function — decoupled from the model class so it
    can be called on any model that accepts (input_ids, use_cache, task)
    and returns (logits, loss).
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        logits, _ = model(generated, use_cache=True, task=task)
        next_logits = logits[:, -1, :].clone() / max(temperature, 1e-9)

        next_logits = apply_repetition_penalty(next_logits, generated[0], repetition_penalty)
        next_logits = top_k_filter(next_logits, top_k)
        next_logits = top_p_filter(next_logits, top_p)

        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_token], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated[0], skip_special_tokens=True)


@torch.no_grad()
def generate_batch(
    model: Any,
    tokenizer: Any,
    prompts: List[str],
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    device: Optional[torch.device] = None,
) -> List[str]:
    """Generate responses for a batch of prompts (greedy for simplicity)."""
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    results = []
    for prompt in prompts:
        out = generate_text(
            model, tokenizer, prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=device,
        )
        results.append(out)
    return results
