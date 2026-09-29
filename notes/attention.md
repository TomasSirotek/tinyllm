5. Attention - attention.py (letting a position look at the past)

	5.1 The problem it solves
	     The bigram sits on `t` and sees ONLY `t`. It cannot tell `...th` from `...righ`.
	     Attention lets a position look back at every earlier letter and decide
	     which ones matter for THIS prediction.

	5.2 First idea: just average the past
	     At each position, use the average of that letter and all letters before it.

		5.2.1 For `hello`:

			position 1 (h):  just h
			position 2 (e):  average of h, e
			position 3 (l):  average of h, e, l
			position 4 (l):  average of h, e, l, l

		5.2.2 Crude (an average throws away order) but it is the first time
		      information from earlier positions reaches a later one.

		5.2.3 THE RULE: position 3 sees positions 1-3 only. Never 4 or 5.
		      Those are the future and the job is to PREDICT the future.
		      Let it peek and it cheats: perfect training loss, useless model.
		      This is called **causal masking**.

	5.3 MY MISTAKE: averaging the wrong direction

		5.3.1 x with B,T,C = 1,5,2 looked like this:

			         col0      col1
			row 0  -2.0260  -2.0655
			row 1  -1.2054  -0.9122
			row 2  -1.2502   0.8032
			row 3  -0.2071   0.0544
			row 4   0.1378  -0.3889

		5.3.2 I tried -2.0260 / -2.0655 / 2 and got 0.49. Two errors:
		      - average is (a + b) / 2. PLUS, not divide.
		      - -2.0260 and -2.0655 are both in ROW 0. They are the two COLUMNS
		        of the SAME position. I averaged sideways.

		5.3.3 Averaging goes DOWN a column, one column at a time:

			row 1, col0:  (-2.0260 + -1.2054) / 2 = -1.6157
			row 1, col1:  (-2.0655 + -0.9122) / 2 = -1.4889
			row 2, col0:  (-2.0260 + -1.2054 + -1.2502) / 3 = -1.4939

		5.3.4 Why columns never mix: C=2 means each position is described by 2
		      numbers = 2 features (say "is vowel" and "is loud"). Averaging
		      vowel-ness with loudness is meaningless. Feature 0 averages with
		      feature 0 across time. Real models have C=384. Same rule.

	5.4 B, T, C - lock these in

		| letter | meaning                                  | mixes?          |
		|--------|------------------------------------------|-----------------|
		| B      | batch - separate examples in parallel    | never mixes     |
		| T      | time - which position in the sequence    | THIS is averaged |
		| C      | channels - the numbers describing one position | never mixes |

	5.5 The matrix multiply trick
	     Averaging is a weighted sum where the weights are equal.
	     So put the weights in a matrix and the whole loop becomes one matmul.

		5.5.1 For T=5:

			row 0   1.00  0.00  0.00  0.00  0.00
			row 1   0.50  0.50  0.00  0.00  0.00
			row 2   0.33  0.33  0.33  0.00  0.00
			row 3   0.25  0.25  0.25  0.25  0.00
			row 4   0.20  0.20  0.20  0.20  0.20

		     Read row 2: take 1/3 each of positions 0,1,2 and ignore 3,4.

		5.5.2 Every row sums to 1 -> that is what makes it an average, not a sum.
		      The zeros in the upper-right triangle ARE the causal mask.

		5.5.3 `torch.tril` = TRIangle Lower. Keeps the diagonal and below, zeroes
		      the rest. Divide each row by its own sum -> 1/(t+1) in each slot.

		5.5.4 `xbow = wei @ x` replaces the whole python loop. One line, runs on
		      the GPU in parallel. Verified with torch.allclose -> True.

		5.5.5 WHY THIS MATTERS: the behaviour is now a matrix of NUMBERS.
		      Numbers can be learned. Nothing forces the row to be 0.33/0.33/0.33.
		      It could be 0.1/0.1/0.8 = "position 2 matters most here".
		      That is attention.

	5.6 Where the weights come from: query, key, value

		5.6.1 Every position produces two things:
		      - a **query**: "what am I looking for?"
		      - a **key**:   "what do I contain?"
		      A vowel might query "I want a consonant before me". A consonant's
		      key says "I am a consonant". They match -> high weight.

		5.6.2 The match score is a **dot product**: multiply elementwise, sum.
		      Big = good match. Negative = mismatch.
		      Do it for every pair of positions -> a T x T grid of scores.
		      That grid is `wei`, now computed from data instead of hardcoded.

		5.6.3 Fix 1 - kill the future: set the upper triangle to **-infinity**.
		      Why -inf and not 0? Softmax comes next and e^(-inf) = 0, so those
		      positions get EXACTLY zero and the rest renormalise among themselves.
		      Using 0 would leave them a share. Silent bug.

		5.6.4 Fix 2 - softmax each row -> weights sum to 1.

		5.6.5 A **value** is what a position hands over when looked at.
		      Separate from the key on purpose: a position can advertise one thing
		      ("I am a consonant") and deliver another (the useful information).
		      query·key decides HOW MUCH. value is WHAT.

	5.7 My run (random, untrained weights)

			row 0:  1.0000  0       0       0       0       0       0      0
			row 3:  0.5792  0.1187  0.1889  0.1131  0       0       0      0
			row 5:  0.0176  0.2689  0.0215  0.0089  0.6812  0.0019  0      0

		5.7.1 Row 5 gives 68% of its attention to position 4 and only 0.19% to
		      ITSELF. It decided another position matters more than its own
		      content. The bigram literally cannot do this.

		5.7.2 Row 3 puts 0.5792 back on position 0 - reaching past its neighbours
		      to the start. **Distance does not matter to attention.** No decay,
		      no locality. That is why transformers beat RNNs at long range.

		5.7.3 Row 0 is 1.0 then zeros - position 0 has no past, no choice to make.

		5.7.4 The pattern is random noise right now because key/query are untrained
		      random layers. Training is what makes it mean something.

	5.8 Left out on purpose (add when training)
	     Real attention divides wei by sqrt(head_size) before the softmax.
	     Without it the scores get large, softmax goes nearly one-hot, and each
	     position attends to exactly ONE predecessor instead of blending.

	5.9 Python gotcha I hit
	     `import torch.nn as nn` does NOT give you the name `torch`.
	     It loads the package but only binds `nn`. Need `import torch` as well,
	     or `torch.manual_seed` fails with NameError.
