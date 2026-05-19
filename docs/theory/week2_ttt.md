# Week 2 — Test-Time Training (TTT) Layers

> **Reading:** Sun et al. (2024), *Learning to (Learn at Test Time): RNNs with Expressive
> Hidden States*, arXiv:2407.04620; Bishop & Bishop ch. 12 (sequence models).

---

## 1. The architectural trilemma

Three families of sequence models confront the same trilemma:

| Family | Time per token | Memory | Expressivity of state |
|---|---|---|---|
| RNN / Mamba | $O(d^2)$ | $O(d)$ | Limited by fixed state size |
| Transformer (full attention) | $O(N d)$ amortized | $O(N d)$ KV-cache | Full token-level recall |
| Test-Time Training (TTT) | $O(d^2)$ | $O(d^2)$ | Linear self-attention as a learned function |

TTT is the conceptual bridge: the state has *high* capacity (a full $d \times d$ matrix), and
it is updated *as if it were a model being trained on the input stream*.

---

## 2. The hidden state as a learned function

Conventional RNNs treat the hidden state $h_t$ as a vector. TTT reinterprets the state as the
parameters of a tiny inner model:

$$
W_t \in \mathbb{R}^{d \times d}, \qquad f_{W_t}(z) = W_t z.
$$

At each time step $t$, the input token $x_t$ produces three projections:

$$
\theta_K(x_t), \quad \theta_V(x_t), \quad \theta_Q(x_t) \in \mathbb{R}^{d},
$$

learned linear maps.

---

## 3. The inner self-supervised loss

The inner model $f_{W_t}$ is trained, **at inference time**, to reconstruct $\theta_V(x_t)$
from $\theta_K(x_t)$:

$$
\ell(W; x_t) \;=\; \tfrac{1}{2}\,\big\| W \theta_K(x_t) - \theta_V(x_t) \big\|_2^2.
$$

A single SGD step with learning rate $\eta$ yields the **update rule**:

$$
W_t = W_{t-1} - \eta \nabla_{W} \ell(W_{t-1}; x_t)
     = W_{t-1} - \eta \big(W_{t-1} \theta_K(x_t) - \theta_V(x_t)\big) \theta_K(x_t)^\top.
$$

This is a **rank-1 update** — computable in $O(d^2)$ and embarrassingly parallelizable across
the sequence dimension when written in unrolled form.

---

## 4. Output and the analogy to linear attention

The TTT output at step $t$ is

$$
y_t = f_{W_t}\!\big(\theta_Q(x_t)\big) = W_t \, \theta_Q(x_t).
$$

When $\eta = 1$ and the initialization is $W_0 = 0$, the closed-form solution of the recurrence
is

$$
W_t = \sum_{s \le t} \theta_V(x_s) \, \theta_K(x_s)^\top,
$$

which is **exactly linear attention with the identity feature map.** TTT therefore strictly
generalizes linear attention: a tunable inner learning rate, multi-step inner optimization,
and non-linear inner models (e.g. an MLP as $f_W$) all become available.

---

## 5. Complexity in practice

For a batch of $B$ sequences of length $N$:

- **Compute:** $O(B \cdot N \cdot d^2)$ — *linear* in $N$, identical to RNN.
- **Memory:** $O(B \cdot d^2)$ for the inner state — independent of $N$.

Compare to full attention's $O(B \cdot N^2 \cdot d)$ time and $O(B \cdot N^2)$ memory. Crossover
occurs at $N \approx d$, which for typical $d = 1024$ means TTT wins decisively for
$N \gtrsim 2k$.

---

## 6. Implementation strategy

Two flavors:

1. **Recurrent form** — straightforward, but cannot be parallelized across $N$.
2. **Mini-batched unrolled form** — process the sequence in chunks of size $k$, take a single
   gradient step per chunk; this is what makes TTT competitive on modern GPUs.

The repository's `src/ttt/ttt_linear.py` implements the recurrent form for clarity and adds
the chunked variant as `TTTLinearChunked`.

---

## 7. Open questions

- **Can a single TTT layer replace several attention layers?** The capstone notebook
  benchmarks this question on TinyStories.
- **Multi-step inner SGD vs. closed-form solution.** For the squared loss, a single Newton step
  solves the inner problem exactly; whether this helps in deep stacks is empirically open.
