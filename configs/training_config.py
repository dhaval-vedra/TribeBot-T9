"""
Named training presets for TribeBot T9.
"""

from __future__ import annotations

from src.tribebot.training.trainer import TrainingConfig


class TrainingPresets:
    """Factory methods for common training configurations."""

    @staticmethod
    def debug() -> TrainingConfig:
        """Minimal config for local smoke testing."""
        return TrainingConfig(
            max_steps=10,
            batch_size=2,
            gradient_accumulation_steps=1,
            eval_interval=5,
            save_interval=10,
            log_interval=1,
            warmup_steps=2,
            lr_decay_steps=10,
            dtype="float32",
        )

    @staticmethod
    def single_gpu() -> TrainingConfig:
        """Standard single-GPU training (A100 40 GB)."""
        return TrainingConfig(
            max_steps=100_000,
            batch_size=8,
            gradient_accumulation_steps=4,
            learning_rate=3e-4,
            warmup_steps=2000,
            dtype="bfloat16",
        )

    @staticmethod
    def multi_gpu() -> TrainingConfig:
        """DDP multi-GPU training with larger effective batch."""
        return TrainingConfig(
            max_steps=200_000,
            batch_size=16,
            gradient_accumulation_steps=2,
            learning_rate=6e-4,
            warmup_steps=5000,
            dtype="bfloat16",
        )
