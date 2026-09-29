3. Bigram model - bigram.py (the simplest thing that is really a language model)

  - Bigram is the thing being tuned. 

  - What the models is - a look-up table like dictionary
  - To generate text - look up current letter, turn those scores into percentages, roll a weighted dice, append the result, repeat. 
  - It is the whole model. 65 keys x 65 scores = 4225 numbers.

Its flaw 
  - it only ever sees one letter
  - when its t is cannot tell if the text so far was right or th 
  - it sees t and nothing else.
  - only produces the letter-pairs that looks good but 
    never the real words!

	3.1 The one-sentence version
	     Look at the current letter. Guess the next letter. That is the whole model.
	     "Bigram" = "two letters": it looks at ONE letter to guess the NEXT one.
	     It has no idea what came before that. Goldfish memory.

	3.2 Worked by hand on one word: `hello`

		3.2.1 Slide a 2-letter window across it

			h e l l o
			└─┘           h -> e
			  └─┘         e -> l
			    └─┘       l -> l
			      └─┘     l -> o

		     Four pairs = four training examples.
		     Left letter is the input, right letter is the answer.

		3.2.2 Count the pairs into a table
		      Rows = the letter I am looking at. Columns = the letter that came next.

			|       | h | e | l | o |
			|-------|---|---|---|---|
			| **h** | 0 | 1 | 0 | 0 |
			| **e** | 0 | 0 | 1 | 0 |
			| **l** | 0 | 0 | 1 | 1 |
			| **o** | 0 | 0 | 0 | 0 |

		     Read row `l`: after an `l` we saw `l` once and `o` once.
		     So sitting on an `l` the model says 50% `l`, 50% `o`.

		3.2.3 That table IS the model. Nothing else.
		      In the real script it is 65x65 because Shakespeare has 65 characters,
		      so 4225 numbers total. Still just a lookup table.

		3.2.4 Generating = start somewhere and keep rolling dice

			h -> row h says 100% e   -> e
			e -> row e says 100% l   -> l
			l -> row l says 50/50    -> l   (dice said l)
			l -> row l says 50/50    -> o   (dice said o)

		     Output `hello`. With different dice: `helllo` or `helo`, and the model is
		     equally happy with those. It never learned the WORD, only the pairs.

	3.3 Logits and softmax

		3.3.1 The model does not store counts, it stores **scores** it tunes by training.
		      Scores can be any number, even negative:

			row 'l' scores:   h: -2.1    e: -1.8    l: 0.9    o: 0.8

		3.3.2 Those raw scores are called **logits**. They are not probabilities —
		      they do not add up to 1.

		3.3.3 **Softmax** squashes them into probabilities that do add to 1:

			softmax ->        h: 0.03    e: 0.04    l: 0.48   o: 0.45

		      Now you can roll dice with them. That is softmax's only job.

	3.4 How it learns: loss

		3.4.1 **Loss = how surprised was the model by the right answer?**
		      Low loss = it expected the correct letter. High loss = blindsided.
		      Training = guess, measure the surprise, nudge the numbers so next time
		      the surprise is smaller. Three thousand times over.

		3.4.2 The reference point: `ln(65) = 4.17`
		      A model that knows nothing and admits it spreads its bet evenly over all
		      65 characters, 1/65 each. The surprise of that is ln(65) = 4.17.
		      You can compute this before running anything.
		      Same trick at any scale: starting loss should be ln(number of choices).

		3.4.3 But we measured **4.73**. Why higher?
		      Because "knows nothing" and "starts at the ignorance loss" are not the same.
		      `nn.Embedding` does not fill the table with zeros, it fills it with RANDOM
		      numbers (mean 0, std 1). So at step 0 the model has random opinions —
		      row `t` might score `q` high for no reason at all.
		      Being uncertain and wrong is cheap. Being CONFIDENTLY wrong is expensive.
		      That penalty is worth about +0.5:

			4.17  even spread, admits ignorance
			4.73  random opinions — what we actually see

		3.4.4 So the diagnostic is a band, not a point

			| first loss              | meaning                                        |
			|-------------------------|------------------------------------------------|
			| 4.2 – 4.8               | correct, carry on                              |
			| much higher (10, 65...) | real bug: targets misaligned or a wrong view() |
			| below 4.17              | impossible without a leak — answer reaching input |

		3.4.5 The lesson hiding in that +0.5: **a random start is worse than knowing
		      nothing.** The model's first job is to unlearn opinions it never earned.
		      Here that wastes a few hundred steps. In a deep network it compounds layer
		      over layer and can stop training working at all — which is why weight
		      initialisation is a real research topic. Comes back at step 5.

	3.5 What good looks like (my run)

			step    0  train 4.7305  val 4.7241
			step  900  train 2.4932  val 2.5088
			step 2700  train 2.4738  val 2.4911

		3.5.1 Settles around **2.45**, train and val nearly equal = no overfitting.
		      4225 numbers cannot memorise a million characters.

		3.5.2 Stops improving around step 900, then just wobbles sideways.
		      That plateau is NOT a bug. It is the model having learned everything a
		      bigram CAN learn. The ceiling is the architecture, not the training.

	3.6 The wall this model hits

		3.6.1 The output has believable letter pairs but no real words.

		3.6.2 Here is exactly why. The model sits on a `t` and is asked what is next.
		      It looks up row `t` — and that row is an average over EVERY `t` in all of
		      Shakespeare. It cannot tell whether the text so far was:

			...th        probably 'e' next -> "the"
			...righ      probably 't' next
			...jus       probably 't' next

		      It only sees `t`. One letter. Everything else is thrown away before it
		      even gets to look.

		3.6.3 **Attention is the fix.** It lets a position look back at all the earlier
		      letters and decide which ones matter for THIS prediction.
		      That is step 5, and it is the one idea the whole transformer is built on.

	3.7 Words worth keeping

		| word            | plain meaning                                          |
		|-----------------|--------------------------------------------------------|
		| token           | one character (for us). Smallest unit the model sees.  |
		| vocab           | the list of all possible tokens. Ours is 65.           |
		| logits          | raw scores, one per vocab entry. Can be negative.      |
		| softmax         | turns logits into probabilities that sum to 1.         |
		| loss            | how surprised the model was by the right answer.       |
		| block_size      | how many tokens of context. 8 for us.                  |
		| batch_size      | how many chunks trained at once, for GPU speed.        |
		| embedding table | the lookup table of scores. For the bigram it IS the model. |
