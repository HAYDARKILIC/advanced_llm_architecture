"""
End-to-end training script for the capstone mini-LLM.

This is a *template* — the dataset loader and the optimizer hyperparameters
are placeholders to be filled in based on the user's environment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from src.model import MiniLLM, MiniLLMConfig
from src.utils.seeding import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the BitWise-LLM-Forge capstone model.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--dataset", type=str, default="tinystories")
    p.add_argument("--quantization", type=str, default="bitnet158")
    p.add_argument("--context-window", type=int, default=4096)
    p.add_argument("--output-dir", type=Path, default=Path("./checkpoints"))
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_model_from_config(path: Path) -> MiniLLM:
    with open(path) as f:
        raw = yaml.safe_load(f)
    # The YAML ``model`` section is intentionally richer than the dataclass
    # (it also carries human-facing metadata such as ``name``, ``attn_impl``,
    # ``norm``, ``ffn``, …). Map the fields the dataclass knows about and
    # silently drop the rest.
    field_map = {
        "vocab_size": "vocab_size",
        "d_model": "d_model",
        "n_layers": "n_layers",
        "n_heads": "n_heads",
        "max_seq_len": "max_seq_len",
        "tie_embeddings": "tie_word_embeddings",
        "tie_word_embeddings": "tie_word_embeddings",
    }
    model_section = raw.get("model", {})
    kwargs = {
        dst: model_section[src]
        for src, dst in field_map.items()
        if src in model_section
    }
    # Optional: derive ffn_mult from d_ff if both d_model and d_ff are given.
    if "d_ff" in model_section and "d_model" in model_section:
        # ffn_mult is the multiplier used in ``MiniLLMConfig``; SwiGLU hidden
        # size in mini_llm.py is ``int(ffn_mult * d_model * 2 / 3)`` rounded
        # up to a multiple of 32. Invert that relation as best we can.
        d = model_section["d_model"]
        d_ff = model_section["d_ff"]
        # Solve d_ff ≈ ffn_mult * d * 2 / 3  →  ffn_mult = round(3 * d_ff / (2 * d))
        kwargs["ffn_mult"] = max(1, round(3 * d_ff / (2 * d)))
    pos_section = raw.get("positional", {})
    if "base_theta" in pos_section:
        kwargs["rope_base"] = float(pos_section["base_theta"])
    cfg = MiniLLMConfig(**kwargs)
    return MiniLLM(cfg)


def build_dataloader(dataset_name: str, ctx: int, batch_size: int) -> DataLoader:
    """TODO: Replace with the real dataset loader (HF datasets, mmap'd tokens, etc.)."""
    # Placeholder: random tokens for the smoke-test path.
    fake_tokens = torch.randint(0, 32_000, (2048, ctx + 1))
    return DataLoader(TensorDataset(fake_tokens), batch_size=batch_size, shuffle=True)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_config(args.config).to(device)
    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    loader = build_dataloader(args.dataset, args.context_window, args.batch_size)

    step = 0
    model.train()
    while step < args.steps:
        for (batch,) in loader:
            batch = batch.to(device)
            ids, targets = batch[:, :-1], batch[:, 1:]
            logits = model(ids)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()

            if step % 50 == 0:
                print(f"step {step:>5d}  loss = {loss.item():.4f}")
            step += 1
            if step >= args.steps:
                break

    ckpt_path = args.output_dir / "mini_llm_final.pt"
    torch.save({"model": model.state_dict(), "config": model.cfg.__dict__}, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
