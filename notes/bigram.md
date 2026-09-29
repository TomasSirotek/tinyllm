# The Bigram Model — plain notes

## The one-sentence version

Look at the current letter. Guess the next letter. That's the whole model.

"Bigram" = "two letters". The prediction looks at **one** letter to guess the **next** one.
It has no idea what came before that. Goldfish memory.

---

## A tiny example by hand

Forget Shakespeare. Our whole world is one word:

```
hello
```

### Step 1: what pairs exist?

Slide a 2-letter window across it:

```
h e l l o
└─┘           h -> e
  └─┘         e -> l
    └─┘       l -> l
      └─┘     l -> o
```

Four pairs. Each pair is one training example: *left letter is the input, right letter is the answer.*

### Step 2: count them in a table

Rows = "the letter I'm looking at". Columns = "the letter that came next".

|       | h | e | l | o |
|-------|---|---|---|---|
| **h** | 0 | 1 | 0 | 0 |
| **e** | 0 | 0 | 1 | 0 |
| **l** | 0 | 0 | 1 | 1 |
| **o** | 0 | 0 | 0 | 0 |

Read row `l`: after an `l`, we saw `l` once and `o` once. So if the model is sitting on
an `l`, it should say "50% chance of `l`, 50% chance of `o`".

**That table is the model.** Nothing else. In the real script the table is 65x65 because
Shakespeare has 65 different characters — 4225 numbers total. Still just a lookup table.

### Step 3: generate

Start on `h`, then keep rolling dice:

```
h -> row h says 100% e     -> e
e -> row e says 100% l     -> l
l -> row l says 50/50      -> l   (dice said l)
l -> row l says 50/50      -> o   (dice said o)
```

Output: `hello`. With different dice you'd get `hellllo` or `helo`, and the model
would be equally happy with those. It never learned the *word* — only the pairs.

---

## Where the words "logits" and "softmax" come in

The model doesn't literally store counts. It stores **scores** that it tunes by training.
Scores can be any number, even negative:

```
row 'l' scores:   h: -2.1    e: -1.8    l: 0.9    o: 0.8
```

These raw scores are called **logits**. They're not probabilities — they don't add to 1.
**Softmax** is the function that squashes them into probabilities that do:

```
softmax ->        h: 0.03    e: 0.04    l: 0.48   o: 0.45
```

Now you can roll dice with them. That's the only job softmax has.

---

## How it learns: loss

**Loss = "how surprised was the model by the right answer?"**

Low loss = it expected the correct letter. High loss = it was blindsided.
Training means: make a guess, measure the surprise, nudge the numbers so next time
the surprise is smaller. Three thousand times over.

### The number that proves it's working

Imagine a model that knows *nothing* and admits it: it spreads its bet evenly across
all 65 characters, 1/65 each. The surprise of that is:

```
ln(65) = 4.17
```

That's the **reference point** — the loss of honest total ignorance. You can compute it
before running anything, and it's the cheapest bug-catcher you'll ever have. Same trick at
any scale: starting loss should be `ln(number of choices)`.

### But we measured 4.73. Why higher?

Because "knows nothing" and "starts at the ignorance loss" aren't the same thing.

`nn.Embedding` doesn't fill the table with zeros. It fills it with **random** numbers
(mean 0, standard deviation 1). So at step 0 the model isn't spreading its bet evenly —
it has random opinions. Row `t` might happen to score `q` high and `h` low, for no reason
at all.

And that costs extra. Being *uncertain* and wrong is cheap; being **confidently wrong** is
expensive. Random opinions make the model confidently wrong about a lot of characters, and
that penalty is worth roughly +0.5 on the loss:

```
4.17  (even spread, admits ignorance)
4.73  (random opinions — what we actually see)
```

So the real diagnostic is a band, not a point:

| first loss | meaning |
|-----------|---------|
| 4.2 – 4.8 | correct. Carry on. |
| much higher (10, 65, ...) | real bug. Usually targets misaligned or a wrong `view()`. |
| below 4.17 | impossible without a leak — the answer is reaching the input somehow. |

There's a lesson hiding in that +0.5: **a random start is worse than knowing nothing.**
The model's first job is to unlearn opinions it never earned. Here it's a trivial waste of
a few hundred steps. In a deep network it compounds layer over layer and can stop training
from working at all — which is why weight initialisation is a real research topic, and
why we'll come back to it at step 5.

### What good looks like

After training on Shakespeare it settles around **2.45**, with train and val nearly equal
(no overfitting — 4225 numbers can't memorise a million characters). It also stops
improving around step 900 and then just wobbles.

That plateau is not a bug. It's the model having learned *everything a bigram can learn*.
The ceiling is the architecture, not the training — which is exactly the point of the
next step.

---

## The wall this model hits

Look at the output it produces. Letter pairs look believable, actual words don't exist.

Here's why, exactly. The model is sitting on a `t` and asked what's next. It looks up
row `t` — and that row is an average over **every** `t` in all of Shakespeare. It cannot
tell whether the text so far was:

```
...th        (probably 'e' next -> "the")
...righ      (probably 't' next)
...jus       (probably 't' next)
```

It only sees `t`. One letter. Everything else is thrown away before it gets to look.

**Attention is the fix.** It lets a position look back at *all* the earlier letters and
decide which ones matter for this particular prediction. That's step 3, and it's the
one idea the entire transformer is built around.

---

## Words worth keeping

| word | plain meaning |
|------|---------------|
| token | one character (for us). The smallest unit the model sees. |
| vocab | the list of all possible tokens. Ours is 65. |
| logits | raw scores, one per vocab entry. Can be negative. |
| softmax | turns logits into probabilities that sum to 1. |
| loss | how surprised the model was by the correct answer. Lower is better. |
| block_size | how many tokens of context. 8 for us. |
| batch_size | how many chunks trained at once, for GPU speed. |
| embedding table | the lookup table of scores. For the bigram, it *is* the model. |
