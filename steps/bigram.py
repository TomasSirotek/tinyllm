from pathlib import Path
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)          # reproducibility: same run, same numbers

batch_size = 32                  # independent chunks processed in parallel
block_size = 8                   # max context length for a prediction
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2             # high, but this model is tiny and convex-ish
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

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Row i = the 65 logits for "what follows token i". The whole model: 65*65 = 4225 numbers.
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx is (B,T) ids -> logits (B,T,C): for every position, a score per vocab entry.
        logits = self.token_embedding_table(idx)

        if targets is None:      # generation path: no targets, so no loss
            return logits, None

        # cross_entropy wants (N,C) logits and (N,) targets, so flatten batch and time together.
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]                  # only the last position predicts what's next
            probs = F.softmax(logits, dim=-1)          # logits -> probabilities
            nxt = torch.multinomial(probs, num_samples=1)   # SAMPLE, don't take the argmax
            idx = torch.cat((idx, nxt), dim=1)         # append and feed back in
        return idx

model = BigramLanguageModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for it in range(max_iters):
    if it % eval_interval == 0:
        losses = estimate_loss(model)
        print(f"step {it:4d}  train {losses['train']:.4f}  val {losses['val']:.4f}")

    xb, yb = get_batch("train")
    _, loss = model(xb, yb)          # forward: compute predictions and loss
    optimizer.zero_grad(set_to_none=True)   # clear last step's grads — PyTorch accumulates otherwise
    loss.backward()                  # backward: gradient of loss w.r.t. every parameter
    optimizer.step()                 # nudge every parameter downhill

print("\n---- sample ----")
start = torch.zeros((1, 1), dtype=torch.long, device=device)   # seed with token 0 ('\n')
print(decode(model.generate(start, max_new_tokens=400)[0].tolist()))
