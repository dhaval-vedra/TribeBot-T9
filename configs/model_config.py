"""
Named model presets for TribeBot T9.
Use these as starting points and adjust as needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.tribebot.model.tribebot import TribeBotConfig


class ModelPresets:
    """Factory methods for common model sizes."""

    @staticmethod
    def debug() -> TribeBotConfig:
        """Tiny model for fast syntax / unit testing on CPU."""
        return TribeBotConfig(
            vocab_size=1000,
            embed_dim=128,
            num_heads=4,
            num_kv_groups=2,
            num_layers=2,
            max_seq_len=512,
            ffn_mult=2,
            dropout=0.0,
            lora_ranks=[2, 4],
        )

    @staticmethod
    def small() -> TribeBotConfig:
        """~350M parameter model — fits on a single A100 40 GB."""
        return TribeBotConfig(
            vocab_size=50_257,
            embed_dim=1024,
            num_heads=16,
            num_kv_groups=4,
            num_layers=24,
            max_seq_len=32_768,
            ffn_mult=4,
        )

    @staticmethod
    def medium() -> TribeBotConfig:
        """~1.3B parameter model."""
        return TribeBotConfig(
            vocab_size=50_257,
            embed_dim=2048,
            num_heads=16,
            num_kv_groups=4,
            num_layers=24,
            max_seq_len=131_072,
            ffn_mult=4,
        )

    @staticmethod
    def large() -> TribeBotConfig:
        """~7B parameter model (T7 scale)."""
        return TribeBotConfig(
            vocab_size=50_257,
            embed_dim=4096,
            num_heads=32,
            num_kv_groups=8,
            num_layers=32,
            max_seq_len=131_072,
            ffn_mult=4,
        )

    @staticmethod
    def ultra() -> TribeBotConfig:
        """~70B parameter model (T8 scale) — requires multi-GPU."""
        return TribeBotConfig(
            vocab_size=50_257,
            embed_dim=8192,
            num_heads=64,
            num_kv_groups=8,
            num_layers=48,
            max_seq_len=262_144,
            ffn_mult=4,
            dropout=0.05,
        )
