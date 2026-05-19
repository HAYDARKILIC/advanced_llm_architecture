<div align="center">

# Advanced LLM Architectures & Optimization

*Forging hardware-aware LLMs from first principles — in pure PyTorch.*

`BitNet b1.58` · `TTT` · `FlashAttention` · `RoPE` · `YaRN` · `Differentiable Logic`

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.3+](https://img.shields.io/badge/PyTorch-2.3+-EE4C2C.svg)](https://pytorch.org/)

</div>

---

Course portfolio for **Advanced LLM Architectures & Optimization** — a six-week,
graduate-level curriculum that re-implements every layer of the modern LLM stack
from first principles. No `nn.MultiheadAttention`, no `transformers` shortcuts.

## Curriculum

| Week | Topic | Notebook |
|:---:|:---|:---|
| 1 | Quantization & ternary weights (BitNet b1.58) | [`01_quantization_bitnet`](notebooks/01_quantization_bitnet.ipynb) |
| 2 | Test-Time Training (TTT) layers | [`02_test_time_training`](notebooks/02_test_time_training.ipynb) |
| 3 | Differentiable logic networks | [`03_differentiable_logic`](notebooks/03_differentiable_logic.ipynb) |
| 4 | FlashAttention — tiling & online softmax | [`04_flash_attention_tiling`](notebooks/04_flash_attention_tiling.ipynb) |
| 5 | RoPE, YaRN & long-context extension | [`05_rope_long_context`](notebooks/05_rope_long_context.ipynb) |
| 6 | Capstone — end-to-end hardware-friendly LLM | [`06_capstone_mini_llm`](notebooks/06_capstone_mini_llm.ipynb) |

Each notebook follows the same structure: mathematical derivation in LaTeX → reference
implementation → sanity checks → optimization → benchmark → discussion. The
companion theory documents live under [`docs/theory/`](docs/theory).

## Quick start

```bash
git clone https://github.com/HAYDARKILIC/advanced_llm_architecture.git
cd bitwise-llm-forge
pip install -r requirements.txt
pip install -r requirements-dev.txt

pytest tests/ -v          # run the unit tests
jupyter lab notebooks/    # open the curriculum
```

## Capstone

The final notebook assembles every component into a working miniature LLM:

```python
from src.model import MiniLLM, MiniLLMConfig

model = MiniLLM(MiniLLMConfig(
    vocab_size=32_000, d_model=384, n_layers=6, n_heads=6, max_seq_len=4_096,
))
```

— ternary `BitLinear` projections, `RotaryPositionalEmbedding`, tiled attention,
SwiGLU FFN, RMSNorm. End-to-end trainable; exports to a packed-ternary `.bwf`
artifact via `scripts/export_quantized.py`.

## Reading

- Raschka, S. *Build a Large Language Model (From Scratch).* Manning, 2024.
- Bishop, C. M. & Bishop, H. *Deep Learning: Foundations and Concepts.* Springer, 2024.

Primary-source papers (BitNet, FlashAttention, RoPE, YaRN, TTT, DiffLogic) are cited inline in each theory document.

---

**Haydar Kılıç** · [@HAYDARKILIC](https://github.com/HAYDARKILIC)
