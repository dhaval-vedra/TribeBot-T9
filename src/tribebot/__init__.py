"""TribeBot T9 — Advanced Reasoning LLM (research package)."""

from .model.tribebot import TribeBotConfig, TribeBotT9
from .model.normalization import RMSNorm, AdvancedRMSNorm
from .model.embeddings import LearnableRoPE, SemanticDynamicVocab
from .model.lora import LoRALayer, MultiRankLoRA
from .model.memory import AdvancedHierarchicalMemory
from .model.attention import GroupedQueryAttention, FFNBlock
from .model.reasoning import (
    RecursiveRefiner,
    InternalDebate,
    WorldModel,
    MetaCognition,
    GraphReasoner,
    CausalInference,
    MathematicalReasoner,
)
from .training.trainer import Trainer, TrainingConfig
from .utils.generation import generate_text, generate_batch

__all__ = [
    "TribeBotConfig",
    "TribeBotT9",
    "RMSNorm",
    "AdvancedRMSNorm",
    "LearnableRoPE",
    "SemanticDynamicVocab",
    "LoRALayer",
    "MultiRankLoRA",
    "AdvancedHierarchicalMemory",
    "GroupedQueryAttention",
    "FFNBlock",
    "RecursiveRefiner",
    "InternalDebate",
    "WorldModel",
    "MetaCognition",
    "GraphReasoner",
    "CausalInference",
    "MathematicalReasoner",
    "Trainer",
    "TrainingConfig",
    "generate_text",
    "generate_batch",
]
