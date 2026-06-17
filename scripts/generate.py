#!/usr/bin/env python3
"""
TribeBot T9 — Text Generation Script
======================================
Usage:
    python scripts/generate.py --checkpoint checkpoints/tribebot_t9_step10000_best.pt \
                               --prompt "Explain quantum entanglement in simple terms."
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoTokenizer

from src.tribebot.model.tribebot import TribeBotT9
from src.tribebot.utils.generation import generate_text
from src.tribebot.utils.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate text with TribeBot T9")
    p.add_argument("--checkpoint",       required=True)
    p.add_argument("--tokenizer",        default="gpt2")
    p.add_argument("--prompt",           required=True)
    p.add_argument("--max_new_tokens",   default=512,  type=int)
    p.add_argument("--temperature",      default=0.8,  type=float)
    p.add_argument("--top_k",            default=50,   type=int)
    p.add_argument("--top_p",            default=0.95, type=float)
    p.add_argument("--rep_penalty",      default=1.1,  type=float)
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    model_cfg = ckpt["config"]
    model = TribeBotT9(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token

    response = generate_text(
        model, tokenizer, args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.rep_penalty,
        device=device,
    )

    print("\n" + "=" * 80)
    print("PROMPT:", args.prompt)
    print("=" * 80)
    print("RESPONSE:", response)
    print("=" * 80)


if __name__ == "__main__":
    main()
