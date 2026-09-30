import torch

from .tokenizer import CharTokenizer


@torch.no_grad()
def generate(model, tokenizer: CharTokenizer, prompt: str = "\n",
             max_new_tokens: int = 500, temperature: float = 1.0,
             top_k: int | None = None) -> str:
    """Continue `prompt` and return the whole thing as text.

    temperature: <1 safer and more repetitive, >1 wilder.
    top_k: only ever sample from the k most likely characters.
    """
    device = next(model.parameters()).device   # put input where the weights already are

    # A char tokenizer only knows the 65 characters it was built from. Anything else
    # would KeyError, so drop unknown characters rather than crash on user input.
    known = [c for c in prompt if c in tokenizer.stoi]
    if len(known) != len(prompt):
        dropped = "".join(sorted(set(prompt) - set(tokenizer.stoi)))
        print(f"[warning] dropped characters not in vocab: {dropped!r}")

    if not known:
        known = ["\n"]                         # empty prompt -> seed with a newline

    idx = torch.tensor([tokenizer.encode("".join(known))], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k)
    return tokenizer.decode(out[0].tolist())
