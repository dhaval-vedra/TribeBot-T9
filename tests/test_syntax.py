"""
Syntax & import smoke tests for TribeBot T9.
Run with:  python tests/test_syntax.py

Test levels
-----------
Level 1 — Syntax   : py_compile on every .py file  (no dependencies needed)
Level 2 — Imports  : importlib.import_module        (requires torch to be installed)
Level 3 — Shapes   : instantiate modules + forward  (requires torch + GPU optional)

Levels 2 & 3 are automatically SKIPPED (not FAILED) when torch is not installed,
because this code is designed for GPU training environments, not CI-only machines.
"""

from __future__ import annotations

import importlib
import py_compile
import sys
import traceback
from pathlib import Path

# Ensure the project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ─────────────────────────────────────────────────────────────
# ANSI colours
# ─────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

PASS  = f"[{GREEN} PASS {RESET}]"
FAIL  = f"[{RED} FAIL {RESET}]"
SKIP  = f"[{YELLOW} SKIP {RESET}]"

# ─────────────────────────────────────────────────────────────
# Check whether torch is available
# ─────────────────────────────────────────────────────────────
try:
    import torch
    TORCH_AVAILABLE = True
    TORCH_VERSION = torch.__version__
except ImportError:
    TORCH_AVAILABLE = False
    TORCH_VERSION = "not installed"


# ─────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────

def check(name: str, fn, skip_if: bool = False, skip_reason: str = ""):
    if skip_if:
        print(f"{SKIP} {name}  ({skip_reason})")
        return None   # None = skipped
    try:
        fn()
        print(f"{PASS} {name}")
        return True
    except Exception:
        print(f"{FAIL} {name}")
        traceback.print_exc()
        return False


# ─────────────────────────────────────────────────────────────
# Level 1 — Syntax (always runs)
# ─────────────────────────────────────────────────────────────

def test_syntax_all_files() -> list:
    python_files = sorted(
        f for f in PROJECT_ROOT.rglob("*.py")
        if "__pycache__" not in str(f)
    )
    results = []
    for fpath in python_files:
        rel = fpath.relative_to(PROJECT_ROOT)
        results.append(check(f"syntax: {rel}", lambda p=fpath: py_compile.compile(str(p), doraise=True)))
    return results


# ─────────────────────────────────────────────────────────────
# Level 2 — Imports (skipped if torch not available)
# ─────────────────────────────────────────────────────────────

MODULES = [
    "src.tribebot.model.normalization",
    "src.tribebot.model.embeddings",
    "src.tribebot.model.lora",
    "src.tribebot.model.memory",
    "src.tribebot.model.reasoning",
    "src.tribebot.model.attention",
    "src.tribebot.model.tribebot",
    "src.tribebot.training.trainer",
    "src.tribebot.utils.generation",
    "src.tribebot.utils.logging_utils",
    "src.tribebot",
]


def test_imports() -> list:
    results = []
    for mod in MODULES:
        results.append(check(
            f"import: {mod}",
            lambda m=mod: importlib.import_module(m),
            skip_if=not TORCH_AVAILABLE,
            skip_reason="torch not installed — install with: pip install torch",
        ))
    return results


# ─────────────────────────────────────────────────────────────
# Level 3 — Component shapes + forward pass
# ─────────────────────────────────────────────────────────────

def test_component_modules() -> list:
    if not TORCH_AVAILABLE:
        print(f"{SKIP} component tests  (torch not installed)")
        return [None]

    import torch
    from src.tribebot.model.normalization import RMSNorm, AdvancedRMSNorm
    from src.tribebot.model.lora import LoRALayer, MultiRankLoRA
    from src.tribebot.model.memory import AdvancedHierarchicalMemory
    from src.tribebot.model.reasoning import (
        RecursiveRefiner, InternalDebate, WorldModel,
        MetaCognition, GraphReasoner, CausalInference, MathematicalReasoner,
    )
    from src.tribebot.model.attention import MultiGateSwiGLU, FFNBlock

    B, T, C = 2, 8, 64

    tests = {
        "RMSNorm":             lambda: RMSNorm(C)(torch.randn(B, T, C)),
        "AdvancedRMSNorm":     lambda: AdvancedRMSNorm(C)(torch.randn(B, T, C)),
        "LoRALayer":           lambda: LoRALayer(C, C, rank=4)(torch.randn(B, T, C)),
        "MultiRankLoRA":       lambda: MultiRankLoRA(C, C, ranks=[2, 4])(torch.randn(B, T, C)),
        "AdvancedHierarchicalMemory": lambda: (
            lambda m, x: (m.add_experience(x), m.retrieve(x))
        )(AdvancedHierarchicalMemory(C, st_capacity=5, mt_capacity=5, lt_capacity=5), torch.randn(B, T, C)),
        "RecursiveRefiner":    lambda: RecursiveRefiner(C)(torch.randn(B, T, C)),
        "InternalDebate":      lambda: InternalDebate(C, num_agents=2)(torch.randn(B, T, C)),
        "WorldModel":          lambda: WorldModel(C)(torch.randn(B, T, C), steps=2),
        "MetaCognition":       lambda: MetaCognition(C)(torch.randn(B, T, C)),
        "GraphReasoner":       lambda: GraphReasoner(C, num_concepts=4)(torch.randn(B, T, C)),
        "CausalInference":     lambda: CausalInference(C)(torch.randn(B, T, C)),
        "MathematicalReasoner":lambda: MathematicalReasoner(C)(torch.randn(B, T, C)),
        "MultiGateSwiGLU":     lambda: MultiGateSwiGLU(C)(torch.randn(B, T, C)),
        "FFNBlock":            lambda: FFNBlock(C, ffn_mult=2)(torch.randn(B, T, C)),
    }
    return [check(f"component: {name}", fn) for name, fn in tests.items()]


def test_model_forward() -> list:
    if not TORCH_AVAILABLE:
        print(f"{SKIP} full model forward pass  (torch not installed)")
        return [None]

    import torch
    from src.tribebot.model.tribebot import TribeBotT9, TribeBotConfig

    def _run():
        cfg = TribeBotConfig(
            vocab_size=200, embed_dim=64, num_heads=4, num_kv_groups=2,
            num_layers=2, max_seq_len=32, ffn_mult=2, dropout=0.0, lora_ranks=[2, 4],
        )
        model = TribeBotT9(cfg)
        model.eval()
        ids     = torch.randint(0, cfg.vocab_size, (1, 16))
        targets = torch.randint(0, cfg.vocab_size, (1, 16))
        with torch.no_grad():
            logits, loss = model(ids, targets=targets)
        assert logits.shape == (1, 16, cfg.vocab_size), f"Wrong logits shape: {logits.shape}"
        assert loss is not None and loss.item() > 0,    "Loss should be positive"

    return [check("full model forward pass (CPU, tiny config)", _run)]


# ─────────────────────────────────────────────────────────────
# Summary helpers
# ─────────────────────────────────────────────────────────────

def _tally(results: list) -> tuple[int, int, int]:
    """Returns (passed, failed, skipped)."""
    passed  = sum(1 for r in results if r is True)
    failed  = sum(1 for r in results if r is False)
    skipped = sum(1 for r in results if r is None)
    return passed, failed, skipped


# ─────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  TribeBot T9 — Smoke Tests")
    print(f"  Python  : {sys.version.split()[0]}")
    print(f"  PyTorch : {TORCH_VERSION}")
    print("=" * 70)

    sections = [
        ("Level 1: Syntax checks (always runs)",       test_syntax_all_files),
        ("Level 2: Import checks (requires torch)",    test_imports),
        ("Level 3: Component shape tests",             test_component_modules),
        ("Level 3: Full model forward pass",           test_model_forward),
    ]

    grand_passed = grand_failed = grand_skipped = 0

    for title, fn in sections:
        print(f"\n{'─' * 70}")
        print(f"  {title}")
        print(f"{'─' * 70}")
        results = fn()
        p, f, s = _tally(results)
        grand_passed  += p
        grand_failed  += f
        grand_skipped += s

    print("\n" + "=" * 70)
    print(f"  RESULTS  passed={grand_passed}  failed={grand_failed}  skipped={grand_skipped}")
    if grand_failed == 0:
        print(f"  {GREEN}ALL TESTS PASSED (or skipped due to missing deps) ✓{RESET}")
    else:
        print(f"  {RED}SOME TESTS FAILED ✗  — see output above{RESET}")
    print("=" * 70 + "\n")

    # Exit 1 only on actual failures, not skips
    sys.exit(0 if grand_failed == 0 else 1)
