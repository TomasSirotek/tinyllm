import os
from pathlib import Path

import requests

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
ROOT = Path(__file__).resolve().parent.parent   # steps/ -> project root
PATH = ROOT / "data" / "input.txt"

if not os.path.exists(PATH):
    print("downloading...")

    os.makedirs(os.path.dirname(PATH), exist_ok=True)

    with open(PATH, "w") as f:
        f.write(requests.get(URL).text)

text = open(PATH).read()

print("characters:", len(text))
print("unique chars:", len(set(text)))
print("---- first 300 ----")
print(text[:300])
