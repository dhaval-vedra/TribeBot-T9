"""
TribeBot T9 — Training Loop
============================
Production-ready training loop with:
  • Gradient accumulation
  • Mixed-precision (torch.amp — not the deprecated torch.cuda.amp)
  • Cosine LR schedule with linear warmup
  • Gradient clipping
  • Checkpoint save / resume
  • Structured logging
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ..model.tribebot import TribeBotT9

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    # Data
    train_data_path: str = "data/train.jsonl"
    val_data_path: str = "data/val.jsonl"
    max_seq_len: int = 2048

    # Optimisation
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    max_grad_norm: float = 1.0

    # Schedule
    warmup_steps: int = 2000
    max_steps: int = 100_000
    lr_decay_steps: int = 100_000

    # Batching
    batch_size: int = 4
    gradient_accumulation_steps: int = 8

    # Evaluation & checkpointing
    eval_interval: int = 500
    eval_iters: int = 100
    save_interval: int = 1000
    checkpoint_dir: str = "checkpoints"

    # Mixed precision
    dtype: str = "bfloat16"    # "float32" | "float16" | "bfloat16"

    # Logging
    log_interval: int = 10
    wandb_project: Optional[str] = None    # Set to enable W&B logging


# ---------------------------------------------------------------------------
# LR Schedule
# ---------------------------------------------------------------------------

def cosine_lr_with_warmup(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    if step < warmup_steps:
        return max_lr * step / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Full training manager for TribeBot T9."""

    def __init__(
        self,
        model: TribeBotT9,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        tokenizer: Any = None,
    ) -> None:
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer

        self.device = next(model.parameters()).device
        self.step = 0
        self.best_val_loss = float("inf")

        # Mixed-precision dtype
        _dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        self.amp_dtype = _dtype_map.get(config.dtype, torch.bfloat16)
        self.use_amp = config.dtype in ("float16", "bfloat16")

        # Gradient scaler (only needed for float16)
        self.scaler = GradScaler(device=str(self.device)) if config.dtype == "float16" else None

        # Optimizer
        self.optimizer = self._build_optimizer()

        # Checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------

    def _build_optimizer(self) -> AdamW:
        """Build AdamW with weight-decay applied only to weight matrices."""
        decay_params, no_decay_params = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim >= 2:
                decay_params.append(param)
            else:
                no_decay_params.append(param)

        groups = [
            {"params": decay_params,    "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        return AdamW(
            groups,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            fused=torch.cuda.is_available(),   # fused kernel when on CUDA
        )

    # ------------------------------------------------------------------
    # LR step
    # ------------------------------------------------------------------

    def _update_lr(self) -> float:
        lr = cosine_lr_with_warmup(
            self.step,
            self.config.warmup_steps,
            self.config.lr_decay_steps,
            self.config.learning_rate,
            self.config.min_lr,
        )
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    # ------------------------------------------------------------------
    # Single gradient-accumulation step
    # ------------------------------------------------------------------

    def _train_step(self, batch: Dict[str, Tensor]) -> float:
        input_ids = batch["input_ids"].to(self.device)
        targets   = batch["labels"].to(self.device)

        acc_steps = self.config.gradient_accumulation_steps
        chunk_size = input_ids.size(0) // acc_steps or 1
        total_loss = 0.0

        for i in range(acc_steps):
            start = i * chunk_size
            end   = min(start + chunk_size, input_ids.size(0))
            if start >= input_ids.size(0):
                break

            ids_chunk = input_ids[start:end]
            tgt_chunk = targets[start:end]

            ctx = autocast(device_type=str(self.device).split(":")[0], dtype=self.amp_dtype) \
                  if self.use_amp else torch.no_grad().__class__()  # plain context manager

            # Use explicit autocast context
            with autocast(device_type=str(self.device).split(":")[0],
                          dtype=self.amp_dtype, enabled=self.use_amp):
                _, loss = self.model(ids_chunk, targets=tgt_chunk)
                loss = loss / acc_steps

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss += loss.item()

        # Gradient clipping + optimiser step
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

        self.optimizer.zero_grad(set_to_none=True)
        return total_loss * acc_steps   # un-normalise for logging

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> float:
        if self.val_loader is None:
            return float("nan")

        self.model.eval()
        losses = []

        for i, batch in enumerate(self.val_loader):
            if i >= self.config.eval_iters:
                break
            input_ids = batch["input_ids"].to(self.device)
            targets   = batch["labels"].to(self.device)
            with autocast(device_type=str(self.device).split(":")[0],
                          dtype=self.amp_dtype, enabled=self.use_amp):
                _, loss = self.model(input_ids, targets=targets)
            losses.append(loss.item())

        self.model.train()
        return sum(losses) / max(len(losses), 1)

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(self, tag: str = "") -> None:
        ckpt_path = Path(self.config.checkpoint_dir) / f"tribebot_t9_step{self.step}{tag}.pt"
        torch.save({
            "step":       self.step,
            "model":      self.model.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "scaler":     self.scaler.state_dict() if self.scaler else None,
            "best_val_loss": self.best_val_loss,
            "config":     self.model.config,
        }, ckpt_path)
        logger.info("Checkpoint saved → %s", ckpt_path)

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.scaler and ckpt.get("scaler"):
            self.scaler.load_state_dict(ckpt["scaler"])
        self.step = ckpt.get("step", 0)
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        logger.info("Checkpoint loaded from %s (step %d)", path, self.step)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        self.model.train()
        logger.info("Starting training | steps=%d | device=%s", self.config.max_steps, self.device)

        loader_iter = iter(self.train_loader)
        t0 = time.time()

        while self.step < self.config.max_steps:
            lr = self._update_lr()

            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(self.train_loader)
                batch = next(loader_iter)

            loss = self._train_step(batch)
            self.step += 1

            # Logging
            if self.step % self.config.log_interval == 0:
                dt = time.time() - t0
                logger.info(
                    "step=%d | loss=%.4f | lr=%.2e | dt=%.2fs",
                    self.step, loss, lr, dt,
                )
                t0 = time.time()

            # Evaluation
            if self.step % self.config.eval_interval == 0:
                val_loss = self.evaluate()
                logger.info("step=%d | val_loss=%.4f", self.step, val_loss)
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("_best")

            # Periodic checkpoint
            if self.step % self.config.save_interval == 0:
                self.save_checkpoint()

        logger.info("Training complete. Best val loss: %.4f", self.best_val_loss)
