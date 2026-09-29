class CharTokenizer:
    """Maps characters to integer ids and back. Reversible, deterministic."""

    def __init__(self, text: str):
        self.chars = sorted(set(text))              # sorted = stable ids across runs
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def state_dict(self) -> dict:
        # the vocab is fully described by the character list - store just that
        return {"chars": "".join(self.chars)}

    @classmethod
    def from_state_dict(cls, d: dict) -> "CharTokenizer":
        # rebuild without needing the original corpus
        return cls(d["chars"])