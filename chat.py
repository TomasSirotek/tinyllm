"""Type a prompt, the model continues it.

This is a base model, not an assistant - it continues text in the style it was
trained on. Try "ROMEO:" rather than "what is 2+2".
"""
from pathlib import Path

import torch

from tinyllm import generate, load_checkpoint

ROOT = Path(__file__).resolve().parent
CKPT = ROOT / "checkpoints" / "model.pt"

TEMPERATURE = 0.6      # <1 safer and more coherent, >1 wilder
TOP_K = 10             # only ever sample from the 10 most likely characters
MAX_NEW_TOKENS = 400


def main() -> None:
    if not CKPT.exists():
        raise SystemExit(f"no checkpoint at {CKPT} - run train.py first")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_checkpoint(CKPT, device)
    params = sum(p.numel() for p in model.parameters())
    print(f"loaded {params:,} parameters on {device}")
    print("type a prompt, or ctrl-c to quit\n")

    while True:
        try:
            prompt = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return

        text = generate(model, tokenizer, prompt or "\n",
                        max_new_tokens=MAX_NEW_TOKENS,
                        temperature=TEMPERATURE,
                        top_k=TOP_K)

        # generate() always stops after exactly MAX_NEW_TOKENS, which lands mid-word.
        # Drop the unfinished last line so the output ends somewhere sensible.
        cut = text.rfind("\n")
        print(text[:cut] if cut > 0 else text)
        print()


if __name__ == "__main__":
    main()
