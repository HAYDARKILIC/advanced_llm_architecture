# Week 1 — Quantization Foundations & Ternary Weights

> **Reading:** Raschka ch. 4 (modeling), Bishop & Bishop ch. 9 (regularization, discretization);
> Ma et al. (2024), *The Era of 1-bit LLMs*, arXiv:2402.17764.

---

## 1. Motivation

A 7B-parameter FP16 model occupies ~14 GB of GPU memory just for weights. Inference latency
is, for autoregressive decoding on modern GPUs, **memory-bandwidth-bound** rather than
compute-bound. Reducing the number of bits per weight therefore directly reduces both the
memory footprint *and* the wall-clock latency, provided the dequantization itself can be
hidden inside the matrix-multiply kernel.

Quantization is the discipline of replacing a high-precision tensor $x \in \mathbb{R}^n$ with
a low-precision encoding $\hat{x} \in \mathcal{Q}^n$ such that the downstream task loss
$\mathcal{L}(f(x; W)) - \mathcal{L}(f(x; \hat{W}))$ is bounded.

---

## 2. PTQ versus QAT

**Post-Training Quantization (PTQ)** treats the trained network as fixed and *projects* each
weight tensor onto the quantization grid. It is cheap (no retraining) but blind to the loss
landscape.

**Quantization-Aware Training (QAT)** simulates the quantization noise *during* training.
The forward pass passes through a non-differentiable rounding operator, and the backward pass
uses the **Straight-Through Estimator (STE)** to propagate gradients as if rounding were the
identity. Formally:

$$
\frac{\partial}{\partial x} \, \text{round}(x) \;\stackrel{\text{STE}}{=}\; 1.
$$

QAT consistently outperforms PTQ in the very-low-bit regime ($\le 4$ bits/weight) because the
optimizer has the chance to redistribute the weight magnitudes so that the rounding error is
small in the directions that matter for the loss.

---

## 3. Absmax (symmetric) quantization

Given a tensor $x$, the symmetric quantizer maps the interval
$[-\max|x|, +\max|x|]$ onto the integer grid $[-(2^{b-1} - 1), +(2^{b-1} - 1)]$:

$$
s = \frac{2^{b-1} - 1}{\max_i |x_i|}, \qquad
q_i = \text{round}(s \cdot x_i), \qquad
\tilde{x}_i = q_i / s.
$$

The maximum quantization error is bounded by $1/(2s)$, hence error decreases linearly with the
number of bits. The symmetric form is preferred for **weights**, whose distribution is
empirically near-zero-mean.

---

## 4. Zero-point (asymmetric) quantization

For tensors with a skewed distribution — typically **activations** after a ReLU or GELU —
an offset $z$ (the *zero-point*) is added:

$$
s = \frac{\max(x) - \min(x)}{2^b - 1}, \qquad
z = -\text{round}\!\left(\frac{\min(x)}{s}\right) - 2^{b-1}, \qquad
q_i = \text{round}(x_i / s) + z.
$$

The asymmetric form recovers a few extra bits of headroom when the underlying distribution is
not symmetric around zero.

---

## 5. BitNet b1.58 — ternary weights

The b1.58 variant restricts weights to $W_{ij} \in \{-1, 0, +1\}$, encoding each weight in
$\log_2(3) \approx 1.58$ bits. The ternarization rule uses the **mean absolute weight**
$\gamma$ as the per-layer scale:

$$
\gamma = \frac{1}{nm} \sum_{i,j} |W_{ij}|, \qquad
W_{\text{tern}} = \text{round}\!\left(\text{clip}\!\left(\frac{W}{\gamma + \epsilon}, -1, +1\right)\right).
$$

The forward pass of a `BitLinear` layer then computes

$$
y = \big( (\text{LayerNorm}(x))_{\text{q8}} \big) \cdot W_{\text{tern}} \cdot \gamma,
$$

where the activation is 8-bit absmax-quantized and the weight matmul reduces to ternary
addition/subtraction (no multiply). On hardware that exploits this, throughput improves by
roughly $1.6\times$–$2.0\times$ over INT8 dense matmuls.

### Scaling laws

Ma et al. (2024) report that, for a fixed FLOP budget, BitNet b1.58 matches the loss of an
FP16 LLaMA-style transformer above ≈3B parameters, with the gap *closing* (not widening) as
model size grows. This is the headline result that motivates the entire field.

---

## 6. Practical notes for implementation

- The Straight-Through Estimator is implemented in PyTorch as
  ```python
  x_quant = (x_quant_no_grad - x).detach() + x
  ```
  This trick passes the quantized values forward but the original gradients backward.
- The activation quantization scale should be computed **per token** (per row of the input),
  not per tensor, to handle outlier tokens gracefully (cf. SmoothQuant).
- `LayerNorm` (or `RMSNorm`) placed *before* the quantized linear is essential — it keeps the
  input distribution stable across training.
