# Changelog

All notable changes to TribeBot T9 are documented here.

---

## [T9.0.0] — 2025

### 🎉 Initial Release — Unified T7 + T8 Architecture

#### Added
- `AdvancedHierarchicalMemory` — 3-scale episodic memory (short / medium / long-term)
- `GraphReasoner` — pure-PyTorch attention-based relational reasoning (replaces networkx)
- `CausalInference` — do-calculus-inspired reasoning with fixed dimension handling
- `MathematicalReasoner` — symbolic encoder + proof checker (fixed tensor indexing)
- `RecursiveRefiner` — iterative quality-critique self-refinement loop (from T7)
- `InternalDebate` — multi-agent consensus mechanism (from T7)
- `WorldModel` — GRU predictive simulation with validity gating (from T7)
- `MetaCognition` — uncertainty-aware processing with per-task goal tracking (from T7)
- `GroupedQueryAttention` — GQA with Flash Attention v2 / PyTorch SDPA fallback
- `LearnableRoPE` — rotary position embeddings with trainable frequency bias
- `MultiRankLoRA` — blends rank-4, 8, 16 LoRA adapters via soft attention weights
- `MultiGateSwiGLU` — enhanced FFN with multiple gating paths
- `AdvancedRMSNorm` — per-element scale + global affine α/β
- `SemanticDynamicVocab` — runtime vocabulary expansion
- Full training loop with gradient accumulation, AMP, cosine LR, checkpointing
- 3-level smoke test suite (syntax / import / forward pass)
- `ModelPresets` — debug / small / medium / large / ultra
- `TrainingPresets` — debug / single_gpu / multi_gpu

#### Fixed (from T7)
- `EnhancedGPTBlock.__init__` was incomplete — all sub-modules now fully defined
- `DynamicVocab` and `RMSNorm` were referenced but never defined
- `WorldModel.simulate` sim_seq blend had wrong dimensions
- `from torch.cuda.amp import autocast` replaced with `from torch.amp import autocast`

#### Fixed (from T8)
- `CausalInference`: `.diag()` on non-square causal_strength matrix
- `MathematicalReasoner.forward`: `mathematical_content[b]` treated Tensor as Python list
- `GraphReasoner`: `networkx` dependency removed; replaced with differentiable PyTorch attention
- `TribeBotT8.generate`: `reasoning_depth` kwarg passed to `forward()` which did not accept it
- `deep_reasoning_cycle` ran all transformer blocks twice — now unified in single forward pass
- `AdvancedHierarchicalMemory.retrieve_relevant_memory`: incorrect batch dimension broadcasting
- `flash_attn_func` called with `[B,H,T,D]` instead of required `[B,T,H,D]` layout
- `scipy` and `faiss` imports removed (were imported but never actually used)

---

## Previous Prototypes

### T8.0 — Ultra Advanced Reasoning AI
- Added: GraphReasoner, CausalInference, MathematicalReasoner
- Added: MultiRankLoRA, MultiGateSwiGLU, LearnableRoPE, SemanticDynamicVocab
- Added: AdvancedHierarchicalMemory (3-scale)
- Status: **Incomplete / buggy** — merged and fixed in T9

### T7.0 — Advanced Human-Level Reasoning AI
- Added: HierarchicalMemory, RecursiveRefiner, InternalDebate, WorldModel, MetaCognition
- Status: **Incomplete / buggy** — merged and fixed in T9
