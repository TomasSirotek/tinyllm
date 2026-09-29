from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
text = (ROOT / "data" / "input.txt").read_text()

chars = sorted(set(text))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}   # string -> int
itos = {i: ch for i, ch in enumerate(chars)}   # int -> string

def encode(s):   return [stoi[c] for c in s]
def decode(ids): return "".join(itos[i] for i in ids)

print("vocab_size:", vocab_size)
print("vocab:", repr("".join(chars)))
print(encode("hello there"))
print(decode(encode("hello there")))

assert decode(encode(text)) == text, "tokenizer is not reversible!"
print("roundtrip OK")

print("tokens:", len(encode(text)), "chars:", len(text))
