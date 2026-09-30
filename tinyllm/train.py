from pathlib import Path

import torch

from .config import Config
from .data import get_batch
from .tokenizer import CharTokenizer

# Only these fields describe the model's shape, so only these go in the checkpoint.
# batch_size or learning_rate can change freely between runs without breaking a load.
ARCH_FIELDS = ("vocab_size", "n_embd", "n_head", "n_layer", "block_size", "dropout")


@torch.no_grad()                    # no gradients here: saves memory and time
def estimate_loss(model, splits, cfg: Config) -> dict:
    """Average the loss over several batches - one batch is far too noisy to read."""
    train_data, val_data = splits
    out = {}
    model.eval()                    # disables dropout: we want a clean measurement
    for name, d in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            X, Y = get_batch(d, cfg)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()                   # back to training mode
    return out


def save_checkpoint(path: Path, model, tokenizer: CharTokenizer, cfg: Config) -> None:
    """Weights alone are useless - the tokenizer and the shape must travel with them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "tokenizer": tokenizer.state_dict(),
        "config": {f: getattr(cfg, f) for f in ARCH_FIELDS},
    }, path)


def train(model, splits, tokenizer: CharTokenizer, cfg: Config, ckpt_path: Path) -> float:
    """Train, checkpointing whenever validation loss improves. Returns the best val loss."""
    train_data, _ = splits
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    best_val = float("inf")

    for it in range(cfg.max_iters):
        if it % cfg.eval_interval == 0 or it == cfg.max_iters - 1:
            losses = estimate_loss(model, splits, cfg)
            marker = ""
            # Save only when val improves. Train loss keeps falling long after the model
            # has started memorising - val is the one that tracks real quality.
            if losses["val"] < best_val:
                best_val = losses["val"]
                save_checkpoint(ckpt_path, model, tokenizer, cfg)
                marker = "  <- saved"
            print(f"step {it:5d}  train {losses['train']:.4f}  val {losses['val']:.4f}{marker}")

        xb, yb = get_batch(train_data, cfg)
        _, loss = model(xb, yb)                 # forward: predictions and loss
        optimizer.zero_grad(set_to_none=True)   # clear last step's grads - they accumulate
        loss.backward()                         # backward: gradient for every parameter
        optimizer.step()                        # nudge every parameter downhill

    print(f"best val loss: {best_val:.4f}  ({ckpt_path})")
    return best_val
