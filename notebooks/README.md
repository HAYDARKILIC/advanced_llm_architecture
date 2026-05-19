# Notebooks Guideline

This document specifies the **exact structure and required content** of every notebook in the
curriculum. Each notebook must follow the seven-section template below.

---

## 📐 Canonical Notebook Template

Every notebook (`01` through `06`) is structured as:

| § | Section | Purpose |
|---|---|---|
| 1 | **Header & Learning Objectives** | Markdown banner with week number, title, prerequisites, and 3–5 concrete learning objectives. |
| 2 | **Mathematical Foundations** | LaTeX-rendered derivations. No code. Reads like a textbook section. |
| 3 | **Reference Implementation** | A minimal, correct implementation in raw PyTorch. Includes type annotations and docstrings. |
| 4 | **Sanity Checks** | Unit-test-style assertions (shape contracts, gradient flow via `torch.autograd.gradcheck`, equivalence to a reference). |
| 5 | **Optimization / Variant** | The "trick" of the week: tiling, ternarization, RoPE rotation, etc. Introduced as a deliberate replacement for a piece of §3. |
| 6 | **Empirical Benchmark** | Quantitative comparison. Memory, latency, perplexity, or accuracy. Results plotted with `matplotlib` and saved to `benchmarks/results/`. |
| 7 | **Discussion & References** | Interpretation of results; pitfalls; the original papers. |

---

## 📓 Notebook-by-Notebook Content Specification

### `01_quantization_bitnet.ipynb`

**Title:** *Week 1 — Quantization Foundations & Ternary Weights (BitNet b1.58)*

**§2 Mathematical Foundations must contain:**
- **PTQ vs. QAT:** loss-landscape sketch; rationale for placing quantization noise inside the
  backward pass.
- **Absmax quantization** (per-tensor):

$$
s = \frac{2^{b-1} - 1}{\max_i |x_i|}, \qquad
\hat{x}_i = \text{round}(s \cdot x_i), \qquad
\tilde{x}_i = \hat{x}_i / s
$$

- **Zero-point (asymmetric) quantization:**

$$
s = \frac{\max(x) - \min(x)}{2^b - 1}, \qquad
z = -\text{round}\!\left(\frac{\min(x)}{s}\right) - 2^{b-1}
$$

- **BitNet b1.58 ternarization rule:**

$$
\gamma = \frac{1}{nm}\sum_{i,j} |W_{ij}|, \qquad
W_{\text{tern}} = \text{round}\!\left(\text{clip}\!\left(\frac{W}{\gamma + \epsilon}, -1, +1\right)\right)
$$

- **Straight-Through Estimator (STE):** forward $= \text{round}(\cdot)$, backward $= \mathbb{1}$.
- **Information-theoretic motivation:** $\log_2(3) \approx 1.58$ bits/parameter.

**§3 Reference Implementation must define:**
- `absmax_quantize(x, n_bits) -> (q, scale)`
- `class BitLinear(nn.Module)` with ternary weight quantization and 8-bit activation quantization
  using STE.

**§5 Optimization variant:** activation pre-LayerNorm placement as in the BitNet paper, and a
RMSNorm fused into the BitLinear forward pass.

**§6 Benchmark deliverables:**
- Table: FP16 vs. INT8-Absmax vs. Ternary on (params, MB, MNIST accuracy).
- Plot: loss vs. training step for the three precisions.

---

### `02_test_time_training.ipynb`

**Title:** *Week 2 — Test-Time Training (TTT) Layers*

**§2 Mathematical Foundations must contain:**
- **Complexity comparison:**

| Layer | Time | Memory |
|---|---|---|
| RNN | $O(N d^2)$ | $O(d)$ |
| Attention | $O(N^2 d)$ | $O(N^2 + N d)$ |
| TTT-Linear | $O(N d^2)$ | $O(d^2)$ |

- **TTT hidden state as a model:** treat the hidden state $W_t \in \mathbb{R}^{d \times d}$ as the
  parameters of an inner model, with an inner self-supervised loss
  $\ell(W; x_t) = \| W \cdot \theta_K(x_t) - \theta_V(x_t) \|_2^2$.
- **Inner update rule:** $W_t = W_{t-1} - \eta \nabla_{W_{t-1}} \ell(W_{t-1}; x_t)$.
- **Closed-form for one SGD step** on the squared loss → a rank-1 update.
- **Comparison with linear attention** as a special case of TTT-Linear with $\eta = 1$.

**§3 Reference Implementation must define:**
- `class TTTLinear(nn.Module)` performing the inner update in a single PyTorch operation
  (vectorized across the sequence dimension).

**§6 Benchmark deliverables:**
- Wall-time vs. sequence length, $N \in \{512, 2048, 8192, 32768\}$, for naïve attention,
  Flash-style attention (using PyTorch's `scaled_dot_product_attention`), and TTT-Linear.
- Peak memory vs. sequence length on the same axes.

---

### `03_differentiable_logic.ipynb`

**Title:** *Week 3 — Differentiable Logic Layers*

**§2 Mathematical Foundations must contain:**
- **The 16 binary boolean gates** ordered as a $16 \times 1$ truth table column.
- **Gate selection as a categorical distribution** with logits $\alpha \in \mathbb{R}^{16}$:

$$
g(a, b) = \sum_{k=1}^{16} \text{softmax}(\alpha)_k \cdot g_k(a, b)
$$

- **Continuous relaxation of the gates,** e.g.
$\text{AND}(a, b) \approx a \cdot b$, $\text{OR}(a, b) \approx a + b - a b$,
$\text{XOR}(a, b) \approx a + b - 2 a b$, and so on for all 16.
- **Smoothing via sigmoid:** map $\mathbb{R} \to (0, 1)$ with $\sigma(\beta z)$ at input;
  increase $\beta$ during training to anneal toward hard binary outputs.
- **Discretization at inference:** $\arg\max_k \alpha_k$.

**§3 Reference Implementation must define:**
- `gate_functions` — a tuple of 16 callables.
- `class DiffLogicLayer(nn.Module)` parameterizing one categorical distribution per output neuron.

**§6 Benchmark deliverables:**
- MNIST accuracy: MLP (baseline) vs. `DiffLogicLayer`, matched parameter count.
- Histogram of selected gates after training (most networks collapse to a handful of gates).

---

### `04_flash_attention_tiling.ipynb`

**Title:** *Week 4 — Ultra-Long Context & Efficient Attention Mechanisms*

**§2 Mathematical Foundations must contain:**
- **Standard attention:**

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d}}\right) V
$$

- **Memory cost of materializing $Q K^\top$:** $O(N^2)$ in HBM.
- **GPU memory hierarchy:** SRAM (≈ 20 MB, ~19 TB/s) vs. HBM (≈ 80 GB, ~2 TB/s) — the
  bandwidth-driven argument for tiling.
- **Online softmax** (Milakov & Gimelshein, 2018):

$$
m^{(j)} = \max(m^{(j-1)}, \tilde{m}^{(j)}), \quad
\ell^{(j)} = e^{m^{(j-1)} - m^{(j)}} \ell^{(j-1)} + \tilde{\ell}^{(j)}
$$

- **FlashAttention tiling:** outer loop over blocks of $Q$, inner loop over blocks of $K, V$;
  $B_r, B_c$ chosen so that each block fits in SRAM.
- **HBM I/O complexity:** $\Theta(N^2 d / M)$ where $M$ is SRAM size — vs. $\Theta(N d + N^2)$ for
  vanilla attention.
- **Linear attention via the kernel trick:**

$$
\text{LinAttn}(Q, K, V) = \phi(Q) \big(\phi(K)^\top V\big), \quad \phi(x) = \text{ELU}(x) + 1
$$

**§3 Reference Implementation must define:**
- `naive_attention(Q, K, V)` — textbook formula, allocates $N \times N$ matrix.
- `tiled_attention(Q, K, V, B_r, B_c)` — pure-PyTorch tiled version using the online softmax
  recurrence. (Not real CUDA — but expresses the algorithm.)
- `linear_attention(Q, K, V)`.

**§6 Benchmark deliverables:**
- Peak memory (allocated by `torch.cuda.max_memory_allocated`) vs. sequence length.
- Wall-time vs. sequence length, with vanilla, tiled, linear, and PyTorch SDPA as baselines.
- Plot demonstrating the $O(N)$ vs. $O(N^2)$ asymptote.

---

### `05_rope_long_context.ipynb`

**Title:** *Week 5 — Advanced Positional Encoding & Context Management*

**§2 Mathematical Foundations must contain:**
- **RoPE as a complex rotation:** pair adjacent dimensions $(x_{2i}, x_{2i+1})$ into a complex
  number $z_i = x_{2i} + i\, x_{2i+1}$; positional encoding multiplies $z_i \cdot e^{i m \theta_i}$
  with $\theta_i = 10000^{-2i/d}$.
- **Relative-position property:**

$$
\langle f(q, m), f(k, n) \rangle = \text{Re}\!\sum_i q_i \bar{k}_i e^{i (m - n) \theta_i}
$$

  which depends only on $m - n$.

- **Why bare RoPE breaks at extrapolation:** high-frequency dimensions ($\theta_i$ large) wrap
  around within the training context length, but low-frequency dimensions are under-trained at
  positions far outside the window.
- **NTK-aware interpolation (YaRN):** scale the base frequency so that high-frequency dimensions
  are preserved and only low-frequency ones are interpolated:

$$
\theta'_i = \theta_i \cdot s^{-2i/(d-2)} \text{ for } i \in [\text{low}, \text{high}]
$$

- **LongRoPE:** evolutionary search over per-dimension scaling factors.

**§3 Reference Implementation must define:**
- `class RotaryPositionalEmbedding(nn.Module)` with cached `cos`/`sin` tables.
- `apply_rotary(x, cos, sin)` helper, fused with the attention `q`/`k` projections.

**§5 Optimization variant:** `class YaRNRotaryPositionalEmbedding(RotaryPositionalEmbedding)`
performing NTK-aware scaling.

**§6 Benchmark deliverables:**
- Train a 6-layer mini-transformer with vanilla RoPE on a 4k-token corpus.
- Evaluate perplexity at context lengths $\{1k, 2k, 4k, 8k, 16k, 32k\}$ for:
  - Vanilla RoPE (no scaling) — expected to explode beyond 4k.
  - Position-interpolated RoPE (PI).
  - YaRN-scaled RoPE.
- Plot perplexity-vs-context-length on one figure.

---

### `06_capstone_mini_llm.ipynb`

**Title:** *Week 6 — End-to-End Hardware-Friendly Mini-LLM (Capstone)*

**§3 Reference Implementation must define:**
- `class MiniLLMBlock(nn.Module)` — transformer block composed of:
  - `RMSNorm` → `BitLinear` projections for Q/K/V → `RotaryPositionalEmbedding` → tiled (or
    linear) attention → `BitLinear` output projection → residual.
  - `RMSNorm` → BitLinear-based SwiGLU MLP → residual.
- `class MiniLLM(nn.Module)` — token embedding + $L$ blocks + final `RMSNorm` + ternary
  output head.

**§5 Optimization variant:** an optional `TTTLinear` block at the bottom of the stack to test
whether a single TTT layer can replace several attention layers under the same parameter budget.

**§6 Benchmark deliverables:**
- Train on TinyStories or WikiText-103 (mini split) for ~5 000 steps on a single GPU.
- Final scoreboard table:

| Model | Params | Bits/param | Disk (MB) | Eval PPL | Latency (ms/tok, CPU) |
|---|---|---|---|---|---|
| FP16 baseline | — | 16 | — | — | — |
| BitWise-LLM-Forge (ternary, RoPE, tiled-attn) | — | 1.58 | — | — | — |

- Save the trained checkpoint to `checkpoints/` and document a `scripts/export_quantized.py`
  command that packs the ternary weights into a compact binary format.

---

## ✅ Style Rules for All Notebooks

- **Determinism.** Every notebook begins with:
  ```python
  from src.utils.seeding import set_seed
  set_seed(42)
  ```
- **Type hints** on every function signature.
- **No magic numbers.** All hyperparameters live in a top-of-notebook `CONFIG` dict, or are loaded
  from `configs/`.
- **Plot styling.** Use a consistent style (e.g. `plt.style.use("seaborn-v0_8-whitegrid")`) and
  always save the resulting PNG to `benchmarks/results/`.
- **Cell tags** for executable-only cells: tag with `nbconvert-skip` if a cell is documentation-only.
- **Notebook outputs are stripped** before commit. CI re-executes them.
