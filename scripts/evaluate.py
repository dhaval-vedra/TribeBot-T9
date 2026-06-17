#!/usr/bin/env python3
"""
TribeBot T9 — Evaluation Script
=================================
Computes perplexity on a held-out dataset.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --data data/val.pt
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader, Dataset

from src.tribebot.model.tribebot import TribeBotT9
from src.tribebot.utils.logging_utils import setup_logging


class TokenDataset(Dataset):
    def __init__(self, data_path: str, seq_len: int) -> None:
        self.data = torch.load(data_path)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx: int) -> dict:
        chunk = self.data[idx : idx + self.seq_len + 1].long()
        return {"input_ids": chunk[:-1], "labels": chunk[1:]}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data",       required=True)
    p.add_argument("--batch_size", default=4, type=int)
    p.add_argument("--max_batches", default=None, type=int)
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

    ds = TokenDataset(args.data, model_cfg.max_seq_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if args.max_batches and i >= args.max_batches:
                break
            ids = batch["input_ids"].to(device)
            lbls = batch["labels"].to(device)
            _, loss = model(ids, targets=lbls)
            n_tokens = (lbls != -100).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(avg_loss)
    print(f"Avg Loss : {avg_loss:.4f}")
    print(f"Perplexity : {perplexity:.2f}")


if __name__ == "__main__":
    main()
