#!/usr/bin/env python3
"""
TribeBot T9 — Training Entry Point
====================================
Usage:
    python scripts/train.py --preset small --data_dir data/ --checkpoint_dir checkpoints/
    python scripts/train.py --preset debug   # quick smoke test

NOTE: This script is for GPU cluster training.
      Do NOT run on Replit — use tests/ for local smoke tests.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader, Dataset

from configs.model_config import ModelPresets
from configs.training_config import TrainingPresets
from src.tribebot.model.tribebot import TribeBotT9
from src.tribebot.training.trainer import Trainer
from src.tribebot.utils.logging_utils import setup_logging


# ---------------------------------------------------------------------------
# Minimal Dataset Stub (replace with your actual dataset)
# ---------------------------------------------------------------------------

class TokenDataset(Dataset):
    """
    Stub dataset that reads pre-tokenised .pt files.
    Each file should be a 1-D LongTensor of token ids.
    Produces (input_ids, labels) pairs for next-token prediction.
    """

    def __init__(self, data_path: str, seq_len: int) -> None:
        p = Path(data_path)
        if p.suffix == ".pt":
            self.data = torch.load(p)
        else:
            raise ValueError(f"Unsupported data format: {p.suffix}. Expected .pt")
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx: int) -> dict:
        chunk = self.data[idx : idx + self.seq_len + 1].long()
        return {
            "input_ids": chunk[:-1],
            "labels":    chunk[1:],
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TribeBot T9")
    p.add_argument("--preset",         default="debug",    choices=["debug", "small", "medium", "large", "ultra"])
    p.add_argument("--train_preset",   default=None,       choices=["debug", "single_gpu", "multi_gpu"])
    p.add_argument("--data_dir",       default="data",     help="Directory containing train.pt and val.pt")
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--resume",         default=None,       help="Path to checkpoint to resume from")
    p.add_argument("--dtype",          default=None,       choices=["float32", "float16", "bfloat16"])
    p.add_argument("--batch_size",     default=None, type=int)
    p.add_argument("--max_steps",      default=None, type=int)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    logger = logging.getLogger(__name__)

    # Model config
    preset_map = {
        "debug":  ModelPresets.debug,
        "small":  ModelPresets.small,
        "medium": ModelPresets.medium,
        "large":  ModelPresets.large,
        "ultra":  ModelPresets.ultra,
    }
    model_cfg = preset_map[args.preset]()

    # Training config
    train_preset = args.train_preset or args.preset
    train_preset_map = {
        "debug":       TrainingPresets.debug,
        "small":       TrainingPresets.single_gpu,
        "medium":      TrainingPresets.single_gpu,
        "large":       TrainingPresets.single_gpu,
        "ultra":       TrainingPresets.multi_gpu,
        "single_gpu":  TrainingPresets.single_gpu,
        "multi_gpu":   TrainingPresets.multi_gpu,
    }
    train_cfg = train_preset_map[train_preset]()
    train_cfg.checkpoint_dir = args.checkpoint_dir
    if args.dtype:
        train_cfg.dtype = args.dtype
    if args.batch_size:
        train_cfg.batch_size = args.batch_size
    if args.max_steps:
        train_cfg.max_steps = args.max_steps

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Model
    model = TribeBotT9(model_cfg).to(device)
    logger.info("Model: %s", model)

    # Data
    train_pt = Path(args.data_dir) / "train.pt"
    val_pt   = Path(args.data_dir) / "val.pt"

    if not train_pt.exists():
        logger.error("Training data not found at %s", train_pt)
        sys.exit(1)

    train_ds = TokenDataset(str(train_pt), model_cfg.max_seq_len)
    val_ds   = TokenDataset(str(val_pt), model_cfg.max_seq_len) if val_pt.exists() else None

    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=train_cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True) if val_ds else None

    # Trainer
    trainer = Trainer(model, train_cfg, train_loader, val_loader)
    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
