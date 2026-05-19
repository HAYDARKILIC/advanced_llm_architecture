# Week 4 — Ultra-Long Context & Efficient Attention

> **Reading:** Dao et al. (2022, 2023), FlashAttention I & II;
> Milakov & Gimelshein (2018), *Online normalizer calculation for softmax*;
> Katharopoulos et al. (2020), *Transformers are RNNs*.

---

## 1. Why vanilla attention is wasteful

Standard scaled dot-product attention computes

$$
\text{Attn}(Q, K, V) \;=\; \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d}}\right) V,
$$

with $Q, K, V \in \mathbb{R}^{N \times d}$. The intermediate matrix $S = Q K^\top \in \mathbb{R}^{N \times N}$
must be **materialized in HBM** for the softmax to operate row-wise. For $N = 32k$ tokens
and FP16 storage, $S$ alone occupies $32{,}768^2 \times 2 = 2$ GB.

The inefficiency is not in FLOPs but in **memory traffic.** Modern GPUs are
bandwidth-bound on attention: writing $S$ to HBM and reading it back for the softmax is the
bottleneck.

---

## 2. The GPU memory hierarchy

| Memory | Size | Bandwidth |
|---|---|---|
| Registers (per thread) | a few KB | ∞ |
| SRAM / shared memory | ~ 100 KB per SM, ~ 20 MB per GPU | ~19 TB/s (A100) |
| HBM | 40–80 GB | ~2 TB/s (A100) |

The optimization principle: **keep intermediate results in SRAM as long as possible.** Every
read/write to HBM costs an order of magnitude more time than the same operation in SRAM.

---

## 3. Online softmax

The softmax over a long vector $z \in \mathbb{R}^N$ can be computed in a single streaming pass.
Maintain a running maximum $m$ and a running denominator $\ell$. For each new block $z^{(j)}$:

$$
\tilde{m}^{(j)} = \max_i z^{(j)}_i, \qquad
\tilde{\ell}^{(j)} = \sum_i e^{z^{(j)}_i - \tilde{m}^{(j)}}.
$$

Update the global statistics by *rescaling*:

$$
m^{(j)} = \max(m^{(j-1)}, \tilde{m}^{(j)}), \qquad
\ell^{(j)} = e^{m^{(j-1)} - m^{(j)}} \ell^{(j-1)} + e^{\tilde{m}^{(j)} - m^{(j)}} \tilde{\ell}^{(j)}.
$$

This recurrence is numerically stable and exact — no need to materialize the whole vector.

---

## 4. FlashAttention — tiled algorithm

FlashAttention applies the online-softmax trick to *both* axes of the attention matrix.

**Outer loop** over $T_r$ blocks of $Q$, of size $B_r \times d$.
**Inner loop** over $T_c$ blocks of $K, V$, of size $B_c \times d$.

For each $Q$-block, the kernel:

1. Loads $Q_i$ into SRAM (once).
2. Iterates over $K_j, V_j$, accumulating the partial output $O_i$ via online softmax — all
   in SRAM.
3. Writes the final $O_i$ back to HBM.

The intermediate $S$ matrix is **never materialized** in HBM. The block sizes $(B_r, B_c)$ are
chosen so that one block of $Q$, one of $K$, one of $V$, and a partial output fit in SRAM:

$$
B_r d + B_c d + B_c d + B_r d \;\le\; M_{\text{SRAM}}.
$$

### I/O complexity

| Algorithm | HBM reads |
|---|---|
| Vanilla attention | $\Theta(N d + N^2)$ |
| FlashAttention | $\Theta(N^2 d^2 / M)$ where $M$ = SRAM size |

For $d = 128$ and SRAM = 100 KB, the FlashAttention bound is roughly $\Theta(N^2 / 800)$,
i.e. a 800× reduction in HBM traffic at the asymptote.

---

## 5. FlashAttention-2

FlashAttention-2 (Dao, 2023) rearranges the outer/inner loop order in the backward pass so
that gradient accumulation matches the forward-pass tiling, eliminating non-coalesced HBM
writes. The result is a $2\times$ speedup on the backward pass — important during training,
where the backward dominates total time.

---

## 6. Linear attention via the kernel trick

If $\phi: \mathbb{R}^d \to \mathbb{R}^{d'}$ is a feature map satisfying
$\phi(q)^\top \phi(k) \approx e^{q^\top k / \sqrt{d}}$, then by associativity:

$$
\text{Attn}(Q, K, V) \approx \phi(Q) \big(\phi(K)^\top V\big).
$$

The right-hand side computes $\phi(K)^\top V \in \mathbb{R}^{d' \times d}$ **first**, producing
a state of size $d' \cdot d$ that does not depend on $N$. This is the **linear attention**
formulation — $O(N)$ time and memory, at the cost of an approximate softmax.

Practical choices for $\phi$:

- $\phi(x) = \text{ELU}(x) + 1$ — original linear attention.
- Performer's random Fourier features.
- $\phi(x) = (\cdot)^2$ component-wise — a simple polynomial kernel.

For autoregressive decoding, the recurrent form is:

$$
S_t = S_{t-1} + \phi(k_t) v_t^\top, \qquad
z_t = z_{t-1} + \phi(k_t), \qquad
y_t = \phi(q_t)^\top S_t \big/ \phi(q_t)^\top z_t.
$$

The state $S \in \mathbb{R}^{d' \times d}$ is constant-size — a true RNN with high-dimensional
state. Compare to TTT in Week 2 — they are deeply related.

---

## 7. KV-cache compression

For inference, the $N \times d$ keys and values dominate memory for long contexts. Compression
techniques (covered briefly in the notebook):

- **Multi-Query Attention** — share K, V across all heads, reducing the cache by a factor of
  $H$ (number of heads).
- **Grouped-Query Attention** — share K, V across groups of heads (compromise).
- **Quantized KV-cache** — 4-bit storage of K, V with online dequant at use time.
- **PagedAttention (vLLM)** — virtual-memory-style block management.
