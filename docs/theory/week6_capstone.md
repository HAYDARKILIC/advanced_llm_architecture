# Week 6 — End-to-End Hardware-Friendly Mini-LLM (Capstone)

> Assembles the techniques of Weeks 1–5 into a single, runnable model.

---

## 1. Design goals

The capstone model satisfies three constraints simultaneously:

- **Sub-2-bit weights.** All linear layers are ternary `BitLinear` (Week 1).
- **Sub-quadratic attention.** Either tiled FlashAttention or linear attention (Week 4).
- **Length-extrapolable.** Rotary embeddings with YaRN scaling (Week 5), trained at 4k tokens
  and evaluated up to 32k.
- *Optional:* A single Test-Time-Training layer (Week 2) as an ablation against attention.

The target footprint is **≤ 100 MB on disk** for a ~30M-parameter model.

---

## 2. Block diagram

```
Token IDs ──┐
            ▼
       Embedding(V, d)
            │
            ▼
   ┌────────────────────────┐
   │  RMSNorm               │
   │  ├── BitLinear(Q,K,V)  │
   │  ├── RotaryEmbedding   │
   │  ├── TiledAttention    │   ← × L blocks
   │  └── BitLinear(O)      │
   │       + residual       │
   │                        │
   │  RMSNorm               │
   │  ├── BitLinear(W1, W3) │   ← SwiGLU
   │  └── BitLinear(W2)     │
   │       + residual       │
   └────────────────────────┘
            │
            ▼
       RMSNorm
            │
            ▼
     BitLinear(d, V)        ← LM head (optionally tied to embedding)
            │
            ▼
         Logits
```

---

## 3. Hyperparameter table

| Hyperparameter | Value |
|---|---|
| Vocabulary size $V$ | 32{,}000 (SentencePiece) |
| Model dim $d$ | 384 |
| Number of layers $L$ | 6 |
| Number of heads $H$ | 6 (head dim 64) |
| MLP expansion | 4 (SwiGLU intermediate = $\tfrac{8}{3} d$) |
| Training context | 4{,}096 |
| Evaluation contexts | $\{1k, 2k, 4k, 8k, 16k, 32k\}$ |
| Optimizer | AdamW, lr $3\times 10^{-4}$, weight decay $0.1$ |
| LR schedule | Cosine, 200 warmup steps |
| Total steps | 5{,}000 |
| Batch size | 32 sequences |
| Precision | FP16 (master weights FP32) for activations; ternary for weights |

Parameter count is approximately:

$$
P \approx V d + L (4 d^2 + 3 d \cdot \tfrac{8}{3} d) + V d \approx 30{,}000{,}000.
$$

At 1.58 bits per weight, the on-disk footprint is

$$
30 \times 10^6 \times 1.58 / 8 \approx 5.9 \text{ MB.}
$$

(Plus FP16 embeddings and norms, which are kept full-precision; final size ≈ 30–40 MB.)

---

## 4. Training pipeline

1. **Tokenization.** Train a SentencePiece BPE on the corpus (TinyStories or WikiText-103-mini).
2. **Pre-tokenize and pack.** Concatenate all tokens, then chunk into 4{,}097-token sequences
   (one extra for next-token targets).
3. **Forward.** All linear layers use `BitLinear`; attention uses tiled attention.
4. **Loss.** Standard cross-entropy over the shifted sequence.
5. **Backward.** STE through ternarization; FP32 master gradients.
6. **Logging.** Track training loss every 50 steps, evaluation perplexity every 500 steps.

---

## 5. Evaluation protocol

Three axes:

1. **Quality.** Token-level perplexity on the held-out validation split, evaluated at each of
   the six context lengths $\{1k, 2k, 4k, 8k, 16k, 32k\}$, with and without YaRN scaling.
2. **Footprint.** Final checkpoint size in MB; verified by `scripts/export_quantized.py`.
3. **Latency.** Per-token wall-clock time on CPU and GPU using a fixed warm-up + median-of-50
   protocol.

The scoreboard is rendered into the final cell of `06_capstone_mini_llm.ipynb` and saved as
`benchmarks/results/capstone_scoreboard.json`.

---

## 6. Ablation matrix

| Variant | Quantization | Attention | Pos. enc. | TTT |
|---|---|---|---|---|
| FP16 baseline | none | naive | learned | — |
| BitWise-LLM (full) | ternary | tiled | RoPE+YaRN | — |
| BitWise + TTT | ternary | tiled (5 blocks) | RoPE+YaRN | 1 block |
| BitWise + linear-attn | ternary | linear | RoPE+YaRN | — |

The notebook produces one row of the table per variant and plots the perplexity-vs-context
curve for each.

---

## 7. Deliverables

After execution of `06_capstone_mini_llm.ipynb`, the following artifacts are produced:

- `checkpoints/mini_llm_final.pt` — full FP32 state dict for further fine-tuning.
- `checkpoints/mini_llm_packed.bwf` — ternary-packed binary format, ~6 MB.
- `benchmarks/results/capstone_scoreboard.{json,png}` — the headline table and figure.
- A short paragraph in the README summarizing the result with absolute numbers.
