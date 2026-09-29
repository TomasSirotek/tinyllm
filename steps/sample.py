"""Generation only - loads the trained checkpoint, no training.

Shows what temperature and top_k actually do to the output.
Run from anywhere:  python steps/sample.py
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))          # so "import tinyllm" works without installing the package

from tinyllm.model import load_checkpoint

device = "cuda" if torch.cuda.is_available() else "cpu"
model, stoi, itos = load_checkpoint(ROOT / "checkpoints" / "model.pt", device)
decode = lambda l: "".join(itos[i] for i in l)

start = torch.zeros((1, 1), dtype=torch.long, device=device)   # seed with token 0 ('\n')

def show(label, **kw):
    print("=" * 70)
    print(label)
    print("=" * 70)
    out = model.generate(start, max_new_tokens=300, **kw)
    print(decode(out[0].tolist()))
    print()

# temperature divides the logits before softmax.
# low  -> the gaps between scores grow  -> the top choice dominates  -> safe, repetitive
# high -> the gaps shrink               -> everything looks equal    -> wild, misspelled
show("temperature 0.5  (timid)",   temperature=0.5)
show("temperature 1.0  (default)", temperature=1.0)
show("temperature 1.5  (unhinged)", temperature=1.5)

# top_k throws away everything except the k most likely characters, THEN samples.
# Stops the model ever picking something absurd, without making it deterministic.
show("temperature 1.0 + top_k 10", temperature=1.0, top_k=10)
