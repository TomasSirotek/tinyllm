from dataclasses import dataclass
import torch

@dataclass
class Config:
    # architecture - these get saved into the checkpoint
    vocab_size: int = 65
    n_embd: int = 384
    n_head: int = 6
    n_layer: int = 6
    block_size: int = 256
    dropout: float = 0.2

    # training - these do not affect the saved model
    batch_size: int = 44 # could be 56 however the difference wont make a difference 
    learning_rate: float = 3e-4 # lowered to 0.0003 - from 1e-2/0.01
    max_iters: int = 4000
    eval_interval: int = 500
    eval_iters: int = 50

    device: str = "cuda" if torch.cuda.is_available() else "cpu"