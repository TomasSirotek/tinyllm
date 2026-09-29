import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)
B, T, C = 4, 8, 32          # bigger now: 4 batches, 8 positions, 32 channels
x = torch.randn(B, T, C)

head_size = 16
key   = nn.Linear(C, head_size, bias=False)   # these three are the learnable parts
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

k = key(x)      # (B,T,16)  what each position contains
q = query(x)    # (B,T,16)  what each position wants
v = value(x)    # (B,T,16)  what each position will share

# every query dotted with every key -> (B,T,T) grid of match scores
wei = q @ k.transpose(-2, -1)

tril = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(tril == 0, float("-inf"))   # block the future
wei = F.softmax(wei, dim=-1)                      # rows now sum to 1

out = wei @ v                                     # collect the values

print("wei[0] =")
print(wei[0])
print("\nout shape:", out.shape)
