7. Sampling - sample.py / chat.py (getting text out)

  - Training is over. Nothing here changes the model.
  - This is only about HOW you pick the next character from the scores.

	7.1 The loop

		7.1.1 Feed in what you have -> get scores for the next character
		      -> pick one -> stick it on the end -> feed it back in.
		      Repeat until you have enough characters.

		7.1.2 Only the LAST position matters. The model gives a prediction for
		      every position, but the earlier ones are predicting characters we
		      already have. Throw them away.

		7.1.3 Crop to the last block_size characters before each step, or the
		      position table runs out of rows.

	7.2 Why sample instead of always taking the best?

		7.2.1 Taking the highest-scoring character every time is deterministic.
		      Same prompt = same output, forever. It also gets stuck in loops,
		      because the safest next character often leads back to where it was.

		7.2.2 torch.multinomial rolls a weighted dice instead. A character with
		      40% probability gets picked about 40% of the time.

	7.3 Temperature

		7.3.1 One division. Divide the scores before softmax.

		7.3.2 LOW (0.5): the gaps between scores get BIGGER, so the top choice
		      dominates. Safe, boring, repetitive.

		7.3.3 HIGH (1.5): the gaps SHRINK, everything starts to look equally
		      likely. Wild, creative, misspelled.

		7.3.4 1.0 = leave the scores as they are.

	7.4 Top-k

		7.4.1 Keep only the k highest-scoring characters, throw the rest away,
		      then sample from what is left.

		7.4.2 Stops the model ever picking something absurd that happened to have
		      a tiny bit of probability - without making it deterministic.

		7.4.3 top_k = 1 means there is nothing to choose, so it IS deterministic.
		      Useful for tests.

		7.4.4 Good default here: temperature 0.8 + top_k 20.

	7.5 -inf again
	     Same trick as the causal mask. To remove a character from the running,
	     set its score to -infinity BEFORE the softmax, because e^(-inf) = 0.
	     Setting it to 0 would leave it with a share of the probability.

	7.6 Two things that must be off during generation

		7.6.1 model.eval() - turns off dropout. Dropout during generation would
		      randomly corrupt the output for no reason.

		7.6.2 torch.no_grad() - we are not learning, so do not build the
		      gradient graph. Saves memory and time.

	7.7 The checkpoint

		7.7.1 Weights alone are useless. Saved together:
		      - the weights
		      - the tokenizer (else id 39 decodes to the wrong character)
		      - the config (else you cannot rebuild the right shape to load into)

		7.7.2 51 MB for 10.8M parameters. Gitignored - weights do not belong in
		      a repo.

	7.8 chat.py

		7.8.1 It is a CONTINUATION, not a conversation.
		      Type "ROMEO:" and get a speech. Ask a question and get nonsense.

		7.8.2 Not a bug to fix. It only ever learned to predict the next
		      character of Shakespeare. It has never seen a question followed by
		      an answer, and 10.8M parameters on 1 MB of text hold no facts.

		7.8.3 Unknown characters get dropped before encoding, because the
		      tokenizer only knows its 65 characters and would otherwise
		      KeyError on anything typed at the prompt.

	7.9 What it actually produced

			> ROMEO:
			Madam, sir.

			ROMEO:
			Three he shall not with him.

	     Real names, real format, real words, no meaning. That is exactly what a
	     base model is.
