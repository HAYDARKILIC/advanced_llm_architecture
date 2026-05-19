# Week 3 — Differentiable Logic Layers

> **Reading:** Petersen et al. (2022), *Deep Differentiable Logic Gate Networks*, NeurIPS;
> Bishop & Bishop ch. 6 (deep networks).

---

## 1. Motivation

Conventional neurons compute $y = \sigma(W x + b)$ — a continuous, smooth function. The
hidden units of a fully trained network often *behave* like discrete decision rules, but the
arithmetic remains floating-point. At inference, a network whose neurons were genuinely
boolean gates would (a) run on integer hardware, (b) be amenable to formal verification, and
(c) reveal an interpretable computation graph.

The challenge: boolean gates are non-differentiable. **Differentiable logic** is a continuous
relaxation that becomes a hard-gate network in the limit.

---

## 2. The 16 binary boolean gates

Two binary inputs admit exactly $2^{2^2} = 16$ functions:

| # | Gate | Formula (real-valued, on $a, b \in [0, 1]$) |
|---|---|---|
| 0 | FALSE | $0$ |
| 1 | AND | $a \cdot b$ |
| 2 | A AND NOT B | $a \cdot (1 - b)$ |
| 3 | A | $a$ |
| 4 | NOT A AND B | $(1 - a) \cdot b$ |
| 5 | B | $b$ |
| 6 | XOR | $a + b - 2 a b$ |
| 7 | OR | $a + b - a b$ |
| 8 | NOR | $1 - (a + b - a b)$ |
| 9 | XNOR | $1 - (a + b - 2 a b)$ |
| 10 | NOT B | $1 - b$ |
| 11 | A OR NOT B | $1 - b + a b$ |
| 12 | NOT A | $1 - a$ |
| 13 | NOT A OR B | $1 - a + a b$ |
| 14 | NAND | $1 - a b$ |
| 15 | TRUE | $1$ |

Each of these is a polynomial in $(a, b)$, hence smooth and differentiable.

---

## 3. Soft gate selection

A `DiffLogicLayer` neuron stores a learnable logit vector
$\alpha \in \mathbb{R}^{16}$. The output is the softmax-weighted sum of the 16 gate outputs:

$$
g_\alpha(a, b) \;=\; \sum_{k=1}^{16} \text{softmax}(\alpha)_k \cdot g_k(a, b).
$$

During training, $\alpha$ is updated by ordinary backpropagation. At inference, the layer is
discretized:

$$
k^* = \arg\max_k \alpha_k, \qquad g(a, b) = g_{k^*}(a, b).
$$

Because each $g_{k^*}$ is a hard boolean gate, the entire inference pass becomes pure integer
logic.

---

## 4. Layer architecture

A `DiffLogicLayer` with $n$ output neurons:

1. Each output neuron $j$ is assigned **two random input indices** $(a_j, b_j)$ drawn once at
   initialization. This sparse wiring is fixed across training — it mirrors the structure of
   real combinational logic circuits.
2. Each neuron computes $g_{\alpha_j}(x_{a_j}, x_{b_j})$.

Stacking $L$ such layers, each of width $n$, yields a network with
$\Theta(L \cdot n)$ parameters — one logit-vector per neuron — and **no matrix multiplications.**

---

## 5. Smoothing and annealing

The piecewise-polynomial gates are continuous on $[0, 1]^2$. To force binary-valued activations
at intermediate layers, a sigmoid with rising sharpness is applied:

$$
\sigma_\beta(z) = \frac{1}{1 + e^{-\beta z}}, \qquad \beta_t = \beta_0 + (\beta_{\max} - \beta_0)\, \frac{t}{T}.
$$

Annealing $\beta$ during training pushes the network toward genuinely binary intermediate
representations, so that the inference-time discretization step does not change predictions.

---

## 6. Theoretical capacity

A network of $L$ differentiable-logic layers with $n$ neurons each is **functionally equivalent**
to a combinational circuit with the same depth and width. By the universality theorem for
boolean circuits, every function $\{0, 1\}^d \to \{0, 1\}^k$ is representable provided
$L$ is large enough. In practice, $L \in [4, 8]$, $n \in [1k, 10k]$ suffices to match MLP
performance on MNIST while running at ~10⁵× lower energy at inference.

---

## 7. Where this fits in the LLM stack

Differentiable logic layers are not (yet) drop-in replacements for transformer FFNs at the
scale of billion-parameter LLMs. The benchmark in `notebooks/03_differentiable_logic.ipynb`
demonstrates the technique on MNIST as a proof of concept and discusses the open research
directions for scaling.
