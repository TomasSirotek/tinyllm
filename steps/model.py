from pathlib import Path
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)          # reproducibility: same run, same numbers

batch_size = 44                  # independent chunks processed in parallel
block_size = 256                  # max context length for a prediction
max_iters = 4000
eval_interval = 500
learning_rate = 3e-4             # lower than the bigram's 1e-2: more parameters, gentler steps
n_embd = 384
n_head = 6
n_layer = 6 
dropout = 0.2
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- data + tokenizer (same as step 1) ----
ROOT = Path(__file__).resolve().parent.parent
text = (ROOT / "data" / "input.txt").read_text()
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join(itos[i] for i in l)

# Encode the whole corpus once into a single flat tensor of ids.
data = torch.tensor(encode(text), dtype=torch.long)

# Held-out validation split. Train loss alone can't tell you memorisation from learning;
# the gap between train and val is how you detect overfitting.
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

def get_batch(split):
    d = train_data if split == "train" else val_data
    # Pick batch_size random start positions. -block_size so a full chunk always fits.
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])          # inputs
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])  # targets = inputs shifted by 1
    return x.to(device), y.to(device)

@torch.no_grad()                 # no gradients needed here: saves memory, and it's faster
def estimate_loss(model, eval_iters=200):
    out = {}
    model.eval()                 # switches off dropout etc. (no-op now, matters later)
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()   # average many batches: a single batch is too noisy to read
    model.train()                # back to training mode
    return out

class Head(nn.Module):
    """one head of self-attention"""
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)   # "what do I contain"
        self.query = nn.Linear(n_embd, head_size, bias=False)   # "what am I looking for"
        self.value = nn.Linear(n_embd, head_size, bias=False)   # "what I hand over"
        # not a parameter, just a constant we need on the right device -> register_buffer
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout) 

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # every query dotted with every key -> (B,T,T) match scores. * C**-0.5 is note 5.8.
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # block the future
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ self.value(x)

class MultiHeadAttention(nn.Module):
    """several heads of self-attention, running in parallel"""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)   # mix the heads back togetheri
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # each head returns (B,T,head_size); glue them along the channel axis -> (B,T,n_embd)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """a simple MLP applied to each position independently"""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),   # widen: more room to compute
            nn.ReLU(),                       # the non-linearity: negatives -> 0, positives unchanged
            nn.Linear(4 * n_embd, n_embd),   # project back down for the residual add
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """transformer block: communication (attention), then computation (ffwd)"""
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa   = MultiHeadAttention(n_head, n_embd // n_head)
        self.ffwd = FeedForward(n_embd)
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)

    def forward(self, x):
        # "x +" is the residual connection: the layer computes a CHANGE to x, not a replacement.
        # That addition is a direct path for gradients to flow back through untouched.
        x = x + self.sa(self.ln1(x))     # normalise, attend, add back
        x = x + self.ffwd(self.ln2(x))   # normalise, think, add back
        return x


class AttentionLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)   # one final norm before the output layer
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_embedding_table(idx)                                    # (B,T,n_embd)
        pos = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T,n_embd)
        x = tok + pos          # "what I am" + "where I am"
        x = self.blocks(x)     # 3 transformer blocks: gather, think, gather, think, ...
        x = self.ln_f(x)       # final normalisation
        logits = self.lm_head(x)                                                 # (B,T,vocab_size)

        if targets is None:
            return logits, None

        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]   # crop: the position table only has block_size rows
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]         # only the last position predicts what's next
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)   # SAMPLE, don't take the argmax
            idx = torch.cat((idx, nxt), dim=1)
        return idx

model = AttentionLanguageModel().to(device)
print("parameters:", sum(p.numel() for p in model.parameters()))
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for it in range(max_iters):
    if it % eval_interval == 0:
        losses = estimate_loss(model)
        print(f"step {it:4d}  train {losses['train']:.4f}  val {losses['val']:.4f}")

    xb, yb = get_batch("train")
    _, loss = model(xb, yb)                 # forward: compute predictions and loss
    optimizer.zero_grad(set_to_none=True)   # clear last step's grads — PyTorch accumulates otherwise
    loss.backward()                         # backward: gradient of loss w.r.t. every parameter
    optimizer.step()                        # nudge every parameter downhill

# SAVE FIRST. Training is the expensive part - never let a bug in the fun part destroy it.
ckpt = ROOT / "checkpoints" / "model.pt"
torch.save({
    "model": model.state_dict(),   # all 10.8M learned numbers
    "stoi": stoi,                  # the tokenizer must travel WITH the model
    "itos": itos,
    "config": dict(n_embd=n_embd, n_head=n_head, n_layer=n_layer,
                   block_size=block_size, vocab_size=vocab_size, dropout=dropout),
}, ckpt)
print("saved:", ckpt)

print("\n---- sample ----")
start = torch.zeros((1, 1), dtype=torch.long, device=device)   # seed with token 0 ('\n')
print(decode(model.generate(start, max_new_tokens=400)[0].tolist()))
