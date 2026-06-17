<div align="center">

# 🤝 Contributing to TribeBot T9

*Thank you for taking the time to contribute! Every contribution makes this project better.*

</div>

---

## 📋 Table of Contents

- [Code of Conduct](#-code-of-conduct)
- [How to Contribute](#-how-to-contribute)
- [Development Setup](#-development-setup)
- [Code Style](#-code-style)
- [Pull Request Process](#-pull-request-process)
- [Reporting Bugs](#-reporting-bugs)
- [Suggesting Features](#-suggesting-features)

---

## 🤝 Code of Conduct

Be respectful, inclusive, and constructive. We welcome contributors of all experience levels.

---

## 💡 How to Contribute

### Types of contributions we love

| Type | Description |
|:-----|:-----------|
| 🐛 **Bug fixes** | Fix errors, wrong shapes, incorrect logic |
| ⚡ **Performance** | Faster attention, better memory usage |
| 🧪 **Tests** | Add test cases for untested components |
| 📖 **Documentation** | Improve docstrings, README, examples |
| 🧠 **New modules** | New reasoning, memory, or attention modules |
| 🌍 **Translations** | Translate README to other languages |

---

## 🛠️ Development Setup

```bash
# 1. Fork the repo on GitHub, then clone your fork
git clone https://github.com/dhaval-vedra/TribeBot-T9.git
cd TribeBot-T9

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install in editable mode with dev dependencies
pip install -e ".[dev]"

# 4. Verify everything works
python tests/test_syntax.py
```

---

## 🎨 Code Style

We follow standard Python conventions:

```bash
# Format code
black src/ tests/ scripts/

# Sort imports
isort src/ tests/ scripts/

# Type checking
mypy src/tribebot/
```

**Key rules:**
- **Type hints** on all function signatures
- **Docstrings** on all public classes and functions
- **No `console.log`** — use Python's `logging` module
- **No silent fallbacks** — raise meaningful errors
- **Comments explain why**, not what

---

## 🔄 Pull Request Process

1. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/my-awesome-feature
   # or
   git checkout -b fix/descriptive-bug-name
   ```

2. **Make your changes** — keep commits small and focused

3. **Run the full test suite**:
   ```bash
   python tests/test_syntax.py
   ```
   All tests must pass (no new failures).

4. **Write a clear PR description** explaining:
   - What changed and why
   - How to test it
   - Any breaking changes

5. **Open the Pull Request** — we'll review within a few days

---

## 🐛 Reporting Bugs

Please include:

```
**Environment**
- Python version:
- PyTorch version:
- CUDA version (if applicable):
- OS:

**Description**
A clear description of the bug.

**Minimal Reproducible Example**
```python
# Paste your code here
```

**Expected behaviour**
What should happen.

**Actual behaviour**
What actually happens (paste the full traceback).
```

---

## 💡 Suggesting Features

Open a GitHub Discussion or Issue with:
- **Problem**: what limitation are you hitting?
- **Proposed solution**: your idea
- **Alternatives**: other approaches you considered
- **References**: papers or repos that inspired the idea

---

<div align="center">

*Happy contributing! ⭐*

</div>
