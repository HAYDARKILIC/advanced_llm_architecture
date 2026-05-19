# Week 5 — Rotary Position Embedding & Context Extension

> **Reading:** Su et al. (2021), *RoFormer*, arXiv:2104.09864; Peng et al. (2023), *YaRN*,
> arXiv:2309.00071; Ding et al. (2024), *LongRoPE*.

---

## 1. The positional-encoding problem

Self-attention is permutation-invariant. To encode token order, one must inject a function
$f(x_m, m)$ of the token $x_m$ and its position $m$, such that the inner product
$\langle f(q, m), f(k, n) \rangle$ depends only on the **relative** offset $m - n$. RoPE is
arguably the most elegant solution: it embeds position as a **rotation** in a complex plane.

---

## 2. RoPE as a complex rotation

Split the $d$-dimensional embedding into $d/2$ pairs $(x_{2i}, x_{2i+1})$, and identify each
pair with a complex number $z_i = x_{2i} + i \, x_{2i+1}$. Define a frequency per pair:

$$
\theta_i = b^{-2i / d}, \qquad b = 10{,}000.
$$

The RoPE map at position $m$ is multiplication by the complex unit $e^{i m \theta_i}$:

$$
f(z_i, m) = z_i \cdot e^{i m \theta_i}.
$$

In real-valued form, this is the standard 2D rotation:

$$
\begin{pmatrix} x'_{2i} \\ x'_{2i+1} \end{pmatrix}
\;=\;
\begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{pmatrix}
\begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}.
$$

Critically, RoPE is **applied only to the queries and keys**, not to the values.

---

## 3. The relative-position property

For two positions $m, n$ and an attention inner product:

$$
\langle f(q, m), f(k, n) \rangle
\;=\; \text{Re} \sum_{i} q_i \, \bar{k}_i \, e^{i (m - n) \theta_i}.
$$

The right-hand side depends **only on the difference** $m - n$. RoPE thereby achieves an
implicit relative positional encoding while remaining a pre-attention transformation — no
$O(N^2)$ relative-position bias matrix is needed.

---

## 4. Why naive RoPE breaks past the training context

Each dimension pair $i$ rotates at angular frequency $\theta_i$. The full period in token
positions is $2\pi / \theta_i$:

- For $i = 0$ (highest frequency): period $\approx 2\pi$ tokens.
- For $i = d/2 - 1$ (lowest frequency): period $\approx 2\pi \cdot b$ tokens
  (≈ 62k for $b = 10{,}000$).

Within the training window $[0, L_{\text{train}}]$:

- **High-frequency dimensions** wrap around many times — the model has seen all phases.
- **Low-frequency dimensions** advance through only a small arc — the model is *untrained*
  on phases beyond $L_{\text{train}} \cdot \theta_i$.

Extrapolating to $N \gg L_{\text{train}}$ exposes the low-frequency dimensions to phases the
model has never seen. Result: perplexity explodes.

---

## 5. Position Interpolation (PI)

Chen et al.'s remedy: rescale positions linearly:

$$
m \mapsto m \cdot \frac{L_{\text{train}}}{L_{\text{target}}}.
$$

Every dimension now stays within the in-distribution range. The cost: the **high-frequency**
dimensions become *less* discriminative — they were already wrapping fine, and we have
unnecessarily compressed their range.

---

## 6. NTK-aware interpolation — the YaRN insight

Peng et al.'s YaRN distinguishes between dimensions by their wavelength:

- Dimensions whose wavelength $\lambda_i = 2\pi / \theta_i$ is **shorter** than the training
  window — *keep them unchanged*.
- Dimensions whose wavelength is **longer** than the training window — *interpolate them
  linearly*.
- A short *interpolation region* in between, where a smooth ramp blends the two regimes.

Formally:

$$
\theta'_i \;=\;
\begin{cases}
\theta_i & \text{if } \lambda_i < L_{\text{train}} / \alpha \\
\theta_i \cdot \frac{L_{\text{train}}}{L_{\text{target}}} & \text{if } \lambda_i > L_{\text{train}} \cdot \beta \\
\text{(smooth blend)} & \text{otherwise}
\end{cases}
$$

with typical $\alpha = 1, \beta = 32$. YaRN also rescales the attention temperature to
compensate for the increased average inner-product magnitude at longer contexts:

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{t \sqrt{d}}\right) V, \quad
t = 0.1 \ln(s) + 1, \quad s = L_{\text{target}} / L_{\text{train}}.
$$

---

## 7. LongRoPE — non-uniform per-dimension scaling

LongRoPE (Ding et al., 2024) drops the assumption that the rescaling factor depends only on
the dimension's wavelength. Instead, it performs an **evolutionary search** over a per-dimension
factor vector $\boldsymbol{s} \in \mathbb{R}^{d/2}$:

$$
\theta'_i = \theta_i \cdot s_i,
$$

optimizing $\boldsymbol{s}$ against perplexity on a held-out long-context corpus. The result is
a non-monotonic factor profile that empirically extends LLaMA-class models to 2M-token contexts
with minimal performance loss.

---

## 8. Implementation notes

- **Cache $\cos$ and $\sin$ tables** of shape $(L_{\text{max}}, d/2)$ on construction; index into
  them at every forward pass.
- **Apply RoPE *after* the Q/K projections** but *before* the attention dot product.
- **Half-precision concerns:** for very long contexts, FP16 accumulation of the rotation can
  drift; compute the cos/sin tables in FP32 and cast at the last moment.
