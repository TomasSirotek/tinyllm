from pathlib import Path
import pytest
from tinyllm import CharTokenizer

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "input.txt"
WORD_HELLO = "hello"
WORD_HELLO_WORD = "hello world"
SENTENCE_FOX = "the quick brown fox"


def test_vocab_size_counts_unique_characters():
    # "hello" has 5 characters but only 4 distinct ones
    tok = CharTokenizer(WORD_HELLO)
    assert tok.vocab_size == 4


def test_roundtrip_preserves_text():
    # The one invariant that must never break: encode then decode is the identity.
    text = CORPUS.read_text()
    tok = CharTokenizer(text)
    assert tok.decode(tok.encode(text)) == text


def test_ids_are_stable_across_instances():
    # sorted() in __init__ is what guarantees this. Without it, set iteration order
    # could shuffle the ids and a saved checkpoint would decode to garbage.
    text = SENTENCE_FOX
    assert CharTokenizer(text).stoi == CharTokenizer(text).stoi


def test_state_dict_roundtrip_rebuilds_same_mapping():
    # Proves a checkpoint can restore the tokenizer without the original corpus.
    tok = CharTokenizer(WORD_HELLO_WORD)
    restored = CharTokenizer.from_state_dict(tok.state_dict())
    assert restored.stoi == tok.stoi
    assert restored.itos == tok.itos
    assert restored.vocab_size == tok.vocab_size


def test_encode_returns_one_id_per_character():
    # True for char-level, and exactly what changes under BPE - so this test is
    # also documentation of which tokenizer we are using.
    tok = CharTokenizer(WORD_HELLO_WORD)
    s = WORD_HELLO
    assert len(tok.encode(s)) == len(s)


def test_unknown_character_raises():
    # Deliberate behaviour: the tokenizer fails loudly. generate() is the layer that
    # filters unknown characters, not this one.
    tok = CharTokenizer(WORD_HELLO)
    with pytest.raises(KeyError):
        tok.encode("z")


def test_decode_of_empty_is_empty():
    tok = CharTokenizer(WORD_HELLO)
    assert tok.decode([]) == ""
