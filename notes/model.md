6. The full model - model.py (putting the pieces together)

  - Attention alone was not enough. Four more things had to be added.
  - Each one was added separately and the loss was checked after each.
  - Nothing here is new maths. It is packaging.

	6.1 Multi-head attention

		6.1.1 One head asks one kind of question. Four heads ask four at once.

		6.1.2 The trick: instead of 1 head with 32 channels, use 4 heads with
		      8 channels each, then glue the outputs back together. 4 x 8 = 32.
		      Same size out, same parameter count.

		6.1.3 The result: 7,553 params both ways. Loss went 2.40 -> 2.27.
		      Free win. Four specialists beat one generalist of the same size.

		6.1.4 nn.ModuleList, not a plain list. A plain [] means PyTorch never
		      sees the parameters, the optimizer never trains them, and nothing
		      errors. It just silently does not learn.

	6.2 Feed-forward

		6.2.1 Attention is COMMUNICATION - positions gathering from each other.
		      Feed-forward is COMPUTATION - each position thinking on its own
		      about what it just gathered.

		6.2.2 Before this, the model gathered and immediately predicted.
		      No thinking step at all.

		6.2.3 It never mixes positions. Every position goes through the same
		      little network alone. The mixing already happened in attention.

		6.2.4 ReLU (negatives -> 0) is what makes it worth having. Two linear
		      layers with nothing between them collapse into one linear layer.
		      The non-linearity is what buys depth.

		6.2.5 Loss 2.27 -> 2.21.

	6.3 Residual connections

		6.3.1 Instead of  x = layer(x)  write  x = x + layer(x)

		6.3.2 So the layer computes a CHANGE to x, not a replacement.

		6.3.3 Why it matters: gradients have to travel from the loss back through
		      every layer, and each layer shrinks and distorts them. The "+" is a
		      direct path - gradients flow straight through an addition untouched.

		6.3.4 Without this, deep networks just do not train. This one trick is
		      most of the reason 100-layer models are possible.

		6.3.5 Most copied idea in modern deep learning. ResNets, transformers,
		      diffusion models - all the same "x +".

	6.4 Layer norm
	     Normalises each position's numbers to mean 0, variance 1 before the layer.
	     Keeps values in a sane range instead of drifting huge or tiny.
	     That drift is what makes deep stacks unstable.

	6.5 The Block = all of it, bundled

			x = x + self.sa(self.ln1(x))     normalise, attend, add back
			x = x + self.ffwd(self.ln2(x))   normalise, think, add back

		6.5.1 Stack 3 of those: 42,369 params, loss 2.07.
		6.5.2 Stack 6 of those and make everything bigger: 10.8M, loss 1.49.

	6.6 Positions need their own embedding

		6.6.1 Attention is an average, and averages do not care about order.
		      "abc" and "cba" would give the exact same result.

		6.6.2 So a second table says "I am position 3" and gets ADDED to the
		      token's numbers.  x = what I am + where I am

		6.6.3 That table has block_size rows, so generate() must crop the context
		      to the last block_size tokens or it crashes.

	6.7 Scaling up (the numbers that actually got to 1.49)

			block_size 8 -> 256      context: 8 characters -> 256
			n_embd    32 -> 384
			n_head     4 -> 6
			n_layer    3 -> 6
			lr      1e-3 -> 3e-4     bigger model, gentler steps
			dropout    0 -> 0.2
			params   42k -> 10,788,929

		6.7.1 12 minutes on the 4050. 3.1 GB of 6 GB VRAM at batch_size 44.

		6.7.2 Dropout = randomly zero 20% of connections each step, so the model
		      cannot lean on one pathway and has to learn redundant patterns.
		      Switched off automatically at eval time by model.eval().

	6.8 Overfitting showed up exactly as predicted

			step 1000:  train 1.6585  val 1.8289   gap 0.17
			step 2000:  train 1.3870  val 1.6113   gap 0.22
			step 3500:  train 1.2295  val 1.5083   gap 0.28
			step 4500:  train 1.1667  val 1.4975   gap 0.33  <- val got WORSE

		6.8.1 Train kept falling, val turned around. From that point on it was
		      memorising this text instead of learning English.

		6.8.2 Best checkpoint was step 4000, not the last one. That is why
		      train() now saves only when val improves.

		6.8.3 10.8M parameters against 1.1M characters - roughly ten numbers per
		      character. Memorising is always waiting. Dropout holds it off.

	6.9 Mistakes I made here

		6.9.1 Added the MultiHeadAttention class but left the line saying
		      Head(n_embd). Loss came out IDENTICAL to four decimals.
		      Lesson: identical loss = identical computation = the new code is
		      not running. Not "the change did not help".

		6.9.2 Pasted new code at the BOTTOM of the file instead of replacing.
		      Python runs top to bottom, so it trained the old model, then
		      defined the new classes, then stopped. Order in a file is not
		      decoration - a name must exist before it is used.

		6.9.3 Pasted a class body and wiped out generate() and the model = line
		      underneath it. Indentation decides what belongs to the class.

		6.9.4 Had torch.save AFTER the sample print. A typo in the print
		      (i400 instead of 400) crashed it and 12 minutes of training was
		      gone. SAVE FIRST. Always.
