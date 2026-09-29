from pathlib import Path

# __file__ is this script's own path; .parent is steps/, .parent.parent is the project root.
# Doing it this way means the path works no matter which directory you run from.
ROOT = Path(__file__).resolve().parent.parent

# Read the entire 1.1M-character corpus into one Python string. Not lines, not words — one stream.
text = (ROOT / "data" / "input.txt").read_text()

# set(text) collapses the corpus to its unique characters (65 of them).
# sorted() makes the order deterministic — critical, because these positions become the token ids.
# Without sorting, set iteration order could shift and a saved model's ids would no longer match.
chars = sorted(set(text))

# How many distinct symbols exist. The model's output layer will have exactly this many neurons:
# at every position it predicts a probability for each possible next character.
vocab_size = len(chars)

# stoi: "string to int" — enumerate pairs each character with its index, so '\n'->0, ' '->1, ... 'z'->64.
# The index is an arbitrary label, not a quantity. 'b' being 40 does not make it one more than 'a'.
stoi = {ch: i for i, ch in enumerate(chars)}

# itos: "int to string" — the same mapping reversed, for turning model output back into readable text.
itos = {i: ch for i, ch in enumerate(chars)}

# Text -> list of ids. One character in, one integer out, so length is preserved exactly.
def encode(s):   return [stoi[c] for c in s]

# Ids -> text. Look each id up and join the characters back into a single string.
def decode(ids): return "".join(itos[i] for i in ids)

print("vocab_size:", vocab_size)
print("vocab:", repr("".join(chars)))
print(encode("hello there"))
print(decode(encode("hello there")))

# The one invariant that must hold: a full round trip over the whole corpus returns the original.
# If this ever fails, every downstream loss number is meaningless — so we fail loudly, here, now.
assert decode(encode(text)) == text, "tokenizer is not reversible!"
print("roundtrip OK")

print("tokens:", len(encode(text)), "chars:", len(text))
