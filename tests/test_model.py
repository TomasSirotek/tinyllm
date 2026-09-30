import math

import pytest
import torch

from tinyllm import CharTokenizer, Config, GPT, load_checkpoint
from tinyllm.train import save_checkpoint

VOCAB = 65
BLOCK = 8


@pytest.fixture
def cfg():
    # Deliberately tiny. The full 10.8M model proves nothing extra and is slow to build.
    # dropout=0 so every test is deterministic - random zeroing would break the mask test.
    return Config(vocab_size=VOCAB, n_embd=32, n_head=4, n_layer=2,
                  block_size=BLOCK, dropout=0.0)


@pytest.fixture
def model(cfg):
    torch.manual_seed(1337)
    m = GPT(cfg)
    m.eval()
    return m


def test_forward_returns_one_score_per_vocab_entry(model):
    idx = torch.randint(0, VOCAB, (2, BLOCK))
    logits, loss = model(idx)
    assert logits.shape == (2, BLOCK, VOCAB)
    assert loss is None                      # no targets given, so nothing to score


def test_forward_with_targets_returns_scalar_loss(model):
    idx = torch.randint(0, VOCAB, (2, BLOCK))
    targets = torch.randint(0, VOCAB, (2, BLOCK))
    _, loss = model(idx, targets)
    assert loss.ndim == 0                    # a single number, not a tensor of them
    assert loss.item() > 0


def test_untrained_loss_is_near_the_ignorance_floor(model):
    # ln(65) = 4.17 is the loss of an even spread over the vocabulary. Random init
    # starts slightly worse (confidently wrong costs more than uncertain), so this
    # is a band, not a point. See notes/bigram.md 3.4.4.
    idx = torch.randint(0, VOCAB, (4, BLOCK))
    targets = torch.randint(0, VOCAB, (4, BLOCK))
    _, loss = model(idx, targets)
    assert math.log(VOCAB) - 0.5 < loss.item() < math.log(VOCAB) + 1.0


def test_generate_appends_exactly_max_new_tokens(model):
    idx = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(idx, max_new_tokens=20)
    assert out.shape == (1, 21)              # the seed token plus 20 new ones


def test_generate_accepts_context_longer_than_block_size(model):
    # The position table only has block_size rows, so generate() must crop.
    # Without that crop this raises an index error.
    idx = torch.randint(0, VOCAB, (1, BLOCK * 3))
    out = model.generate(idx, max_new_tokens=5)
    assert out.shape == (1, BLOCK * 3 + 5)


def test_top_k_of_one_is_deterministic(model):
    # top_k=1 leaves a single candidate, so sampling has no choice to make.
    idx = torch.zeros((1, 1), dtype=torch.long)
    a = model.generate(idx, max_new_tokens=15, top_k=1)
    b = model.generate(idx, max_new_tokens=15, top_k=1)
    assert torch.equal(a, b)


def test_attention_cannot_see_the_future(model):
    """The most important test here.

    A broken causal mask lets each position peek at the answer. Training loss looks
    fantastic, generation is garbage, and nothing raises. So: change the LAST token
    and assert every EARLIER position's logits are untouched.
    """
    idx = torch.randint(0, VOCAB, (1, BLOCK))
    changed = idx.clone()
    changed[0, -1] = (idx[0, -1] + 1) % VOCAB     # perturb only the final position

    before, _ = model(idx)
    after, _ = model(changed)

    # every position before the change must be bit-for-bit unaffected
    assert torch.allclose(before[:, :-1], after[:, :-1], atol=1e-6)
    # and the changed position itself must actually differ, or the test proves nothing
    assert not torch.allclose(before[:, -1], after[:, -1], atol=1e-6)


def test_checkpoint_roundtrip_reproduces_the_same_logits(model, cfg, tmp_path):
    # Weights alone are not a checkpoint: the tokenizer and the shape must travel too.
    tok = CharTokenizer("".join(chr(32 + i) for i in range(VOCAB)))
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, tok, cfg)

    restored, restored_tok = load_checkpoint(path)

    idx = torch.randint(0, VOCAB, (1, BLOCK))
    assert torch.allclose(model(idx)[0], restored(idx)[0], atol=1e-6)
    assert restored_tok.stoi == tok.stoi
