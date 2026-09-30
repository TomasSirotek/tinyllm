from pathlib import Path
from tinyllm import Config, CharTokenizer, GPT
from tinyllm.data import load_corpus, make_splits
from tinyllm.train import train

ROOT = Path(__file__).resolve().parent

cfg = Config()                              # all defaults from the dataclass
text = load_corpus(ROOT / "data" / "input.txt") 

tok = CharTokenizer(text)                   # builds the 65-char vocab

cfg.vocab_size = tok.vocab_size             # config learns the vocab size from the data

splits = make_splits(text, tok)             # encodes and splits 90/10
model = GPT(cfg).to(cfg.device)

train(model, splits, tok, cfg,
      ROOT / "checkpoints" / "model.pt",
      ROOT / "checkpoints" / "history.csv")   # loss curve, for docs/plot_loss.py