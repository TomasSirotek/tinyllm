from pathlib import Path

import requests
import torch
from torch import Tensor

from .config import Config

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def load_corpus(path: Path) -> str:
    """Return the corpus as one string, downloading it first if needed."""
    if not path.exists():
        print("downloading...")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(requests.get(URL).text)

    return path.read_text()          # always runs, downloaded or not


def make_splits(text: str, tokenizer, train_frac: float = 0.9) -> tuple[Tensor, Tensor]:
    """Encode the whole corpus once, then cut it into train and validation."""
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(train_frac * len(data))
    return data[:n], data[n:]


def get_batch(data: Tensor, cfg: Config) -> tuple[Tensor, Tensor]:
    """Sample batch_size random chunks. y is x shifted one position to the left."""
    # -block_size so a full chunk plus its shifted target always fits
    ix = torch.randint(len(data) - cfg.block_size, (cfg.batch_size,))
    x = torch.stack([data[i:i + cfg.block_size] for i in ix])
    y = torch.stack([data[i + 1:i + cfg.block_size + 1] for i in ix])
    return x.to(cfg.device), y.to(cfg.device)