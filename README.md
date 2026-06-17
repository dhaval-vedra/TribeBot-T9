<p align="center">
  <img src="assets/banner.svg" alt="Banner" width="100%"/>
</p>

# TribeBot-T9 🚀

<!-- Badges -->
<p align="center">
  <a href="https://github.com/dhaval-vedra/TribeBot-T9/stargazers">
    <img src="https://img.shields.io/github/stars/dhaval-vedra/TribeBot-T9?style=for-the-badge&logo=github&color=a78bfa&labelColor=0f0c29" alt="Stars"/>
  </a>
  <a href="https://github.com/dhaval-vedra/TribeBot-T9/forks">
    <img src="https://img.shields.io/github/forks/dhaval-vedra/TribeBot-T9?style=for-the-badge&logo=github&color=60a5fa&labelColor=0f0c29" alt="Forks"/>
  </a>
  <a href="https://github.com/dhaval-vedra/TribeBot-T9/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-34d399?style=for-the-badge&labelColor=0f0c29" alt="License"/>
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/Python-3.10%2B-f472b6?style=for-the-badge&logo=python&logoColor=white&labelColor=0f0c29" alt="Python"/>
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/PyTorch-2.1%2B-fbbf24?style=for-the-badge&logo=pytorch&logoColor=white&labelColor=0f0c29" alt="PyTorch"/>
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/Status-Research-c084fc?style=for-the-badge&labelColor=0f0c29" alt="Status"/>
  </a>
</p>

<p align="center">
  <img src="assets/logo.svg" alt="TribeBot T9 Logo" width="160"/>
</p>

<p align="center">
  <em><b>The most feature-rich open-source reasoning LLM architecture — built for researchers who want everything in one place.</b></em>
</p>

---

## ✨ What Makes T9 Different?

Most open-source LLM codebases give you **just a transformer**. TribeBot T9 ships **7 reasoning systems**, **3-scale memory**, **multi-rank LoRA**, **Graph + Causal + Mathematical reasoning** — all merged, all fixed, all working on CPU or GPU out of the box.

> 💡 This is a **research prototype** — designed for academics, AI enthusiasts, and engineers who want to study and experiment with advanced LLM reasoning architectures.

---

## 📋 Table of Contents

- [✨ What Makes T9 Different?](#-what-makes-t9-different)
- [🚀 Features](#-features)
- [🏗️ Architecture](#️-architecture)
- [📁 Project Structure](#-project-structure)
- [⚡ Quick Start](#-quick-start)
- [🎛️ Model Presets](#️-model-presets)
- [🔧 Configuration](#-configuration)
- [🧪 Testing](#-testing)
- [🏋️ Training](#️-training)
- [💬 Text Generation](#-text-generation)
- [📊 Performance](#-performance)
- [🗺️ Roadmap](#️-roadmap)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [📚 Citation](#-citation)

---

## 🚀 Features

<table>
<tr>
<td width="50%" valign="top">

### 🧠 Advanced Reasoning Stack
- **Recursive Self-Refinement** — quality-critique loop that iteratively improves outputs
- **Internal Multi-Agent Debate** — 3 parallel agents reach consensus before producing output
- **World Model** — GRU-based predictive simulation with validity gating
- **MetaCognition** — uncertainty-aware self-regulation with per-task goal tracking

</td>
<td width="50%" valign="top">

### 🔬 Structural Reasoning
- **Graph Reasoner** — pure-PyTorch attention-based relational reasoning (zero external deps)
- **Causal Inference** — do-calculus inspired reasoning, fixed Pearl's intervention operator
- **Mathematical Reasoner** — symbolic encoder + proof checker for math-heavy tasks

</td>
</tr>
<tr>
<td valign="top">

### 💾 3-Scale Episodic Memory
- **Short-term** — raw experience vectors (fast, 50-item buffer)
- **Medium-term** — transformer-encoded summaries (200-item buffer)
- **Long-term** — GRU-consolidated memories (500-item buffer)
- Cross-attention based retrieval — models attend to relevant past context

</td>
<td valign="top">

### ⚡ Production Attention
- **Grouped Query Attention (GQA)** — fewer KV heads → lower memory at scale
- **Learnable RoPE** — rotary position embeddings with trainable frequency bias
- **Flash Attention v2** — optional GPU kernel, auto-fallback to PyTorch SDPA
- **KV Cache** — fast autoregressive decoding

</td>
</tr>
<tr>
<td valign="top">

### 🎛️ Adaptation & Efficiency
- **Multi-Rank LoRA** — blends rank-4, 8, 16 adapters via learned soft attention weights
- **Multi-Gate SwiGLU FFN** — extra gating paths for richer feed-forward dynamics
- **Advanced RMSNorm** — per-element scale + global affine α/β parameters
- **Semantic Dynamic Vocab** — runtime vocabulary expansion with semantic init

</td>
<td valign="top">

### 🛠️ Training Infrastructure
- Mixed precision (`bfloat16 / float16 / float32`) — modern `torch.amp` API
- Gradient accumulation + gradient clipping
- Cosine LR schedule with linear warmup
- Checkpoint save / resume
- Optional Weights & Biases integration

</td>
</tr>
</table>

---

## 🏗️ Architecture
```
┌─────────────────────────────────────────┐
│          Input Token IDs [B, T]         │
└───────────────────┬─────────────────────┘
│
┌───────────────────▼─────────────────────┐
│  SemanticDynamicVocab + PosEmbedding    │
└───────────────────┬─────────────────────┘
│
╔═══════════════════▼═════════════════════╗
║          TribeBotBlock  × N             ║
║                                         ║
║  ┌─────────────────────────────────┐   ║
║  │   AdvancedHierarchicalMemory     │   ║
║  │   Short ──► Medium ──► Long-term │   ║
║  └──────────────┬──────────────────┘   ║
║                 │                        ║
║  ┌──────────────▼──────────────────┐   ║
║  │         GraphReasoner            │   ║
║  │  (pure-PyTorch message passing)  │   ║
║  └──────────────┬──────────────────┘   ║
║                 │                        ║
║  ┌──────────────▼──────────────────┐   ║
║  │        CausalInference           │   ║
║  │   (do-calculus, interventions)   │   ║
║  └──────────────┬──────────────────┘   ║
║                 │                        ║
║  ┌──────────────▼──────────────────┐   ║
║  │   GroupedQueryAttention + RoPE   │   ║
║  │      + MultiRankLoRA adapters    │   ║
║  └──────────────┬──────────────────┘   ║
║                 │                        ║
║  ┌──────────────▼──────────────────┐   ║
║  │    FFNBlock (MultiGateSwiGLU)    │   ║
║  └──────────────┬──────────────────┘   ║
║                 │                        ║
║  ┌──────────────▼──────────────────┐   ║
║  │   MathematicalReasoner           │   ║
║  └──────────────┬──────────────────┘   ║
╚═══════════════════▼═════════════════════╝
│
┌─────────────────────────────▼──────────────────────────────┐
│                  High-Level Reasoning Stack                 │
│  InternalDebate → RecursiveRefiner → MetaCognition → WorldModel │
└─────────────────────────────┬──────────────────────────────┘
│
┌───────────────────▼─────────────────────┐
│       AdvancedRMSNorm  →  LM Head       │
└───────────────────┬─────────────────────┘
│
┌───────────────────▼─────────────────────┐
│          Logits / Loss [B, T, V]        │
└─────────────────────────────────────────┘
```

